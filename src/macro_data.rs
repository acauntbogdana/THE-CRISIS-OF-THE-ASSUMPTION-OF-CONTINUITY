// Загрузка макроэкономических данных.

use std::{
    collections::{btree_map::Entry, BTreeMap, HashMap},
    fs::File,
};

use anyhow::{anyhow, Context, Result};
use arrow::{
    array::{Array, ArrayRef, Float64Array, TimestampMicrosecondArray},
    compute::cast,
    datatypes::{DataType, TimeUnit},
};
use chrono::{DateTime, NaiveDate};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use crate::config::MACRO_FILE;

pub fn load_macro_ffill() -> Result<BTreeMap<NaiveDate, HashMap<String, Option<f64>>>> {
    let file = File::open(MACRO_FILE).context("не удалось открыть макро-файл")?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .context("не удалось создать builder для parquet")?;
    let schema = builder.schema().clone();
    let reader = builder
        .with_batch_size(1000)
        .build()
        .context("не удалось построить reader")?;

    // Проверяем, что колонка с именем "date" ровно одна
    let date_indices: Vec<usize> = schema
        .fields()
        .iter()
        .enumerate()
        .filter(|(_, f)| f.name() == "date")
        .map(|(i, _)| i)
        .collect();
    if date_indices.is_empty() {
        return Err(anyhow!("в макро-файле отсутствует колонка 'date'"));
    }
    if date_indices.len() > 1 {
        return Err(anyhow!(
            "в макро-файле найдено несколько колонок с именем 'date'"
        ));
    }
    let idx_date = date_indices[0];

    // Собираем все колонки с суффиксом "_ffill" (исключая саму дату)
    let indices: Vec<(String, usize)> = schema
        .fields()
        .iter()
        .enumerate()
        .filter(|(i, f)| *i != idx_date && f.name().ends_with("_ffill"))
        .map(|(i, f)| (f.name().clone(), i))
        .collect();

    if indices.is_empty() {
        return Err(anyhow!(
            "в макро-файле не найдено ни одной колонки с суффиксом _ffill"
        ));
    }

    let mut map: BTreeMap<NaiveDate, HashMap<String, Option<f64>>> = BTreeMap::new();

    for batch in reader {
        let batch = batch.context("ошибка чтения батча")?;
        if batch.num_rows() == 0 {
            continue;
        }

        // Приводим колонку даты к TimestampMicrosecondArray
        let date_array = cast(
            batch.column(idx_date),
            &DataType::Timestamp(TimeUnit::Microsecond, None),
        )
        .with_context(|| {
            format!(
                "не удалось привести колонку 'date' (тип {:?}) к TimestampMicrosecond",
                batch.column(idx_date).data_type()
            )
        })?;
        let date_arr = date_array
            .as_any()
            .downcast_ref::<TimestampMicrosecondArray>()
            .expect("после cast тип должен быть TimestampMicrosecondArray");

        // Приводим все ffill-колонки к Float64Array, сохраняя ArrayRef,
        // чтобы они жили достаточно долго.
        let mut casted_arrays: Vec<ArrayRef> = Vec::with_capacity(indices.len());
        for (_name, idx) in &indices {
            let col = batch.column(*idx);
            let col_casted = cast(col, &DataType::Float64).with_context(|| {
                format!(
                    "не удалось привести колонку (тип {:?}) к Float64",
                    col.data_type()
                )
            })?;
            casted_arrays.push(col_casted);
        }

        let mut float_cols: Vec<(&str, &Float64Array)> = Vec::with_capacity(indices.len());
        for (j, (name, _)) in indices.iter().enumerate() {
            let float_arr = casted_arrays[j]
                .as_any()
                .downcast_ref::<Float64Array>()
                .expect("после cast тип должен быть Float64Array");
            float_cols.push((name.as_str(), float_arr));
        }

        // Обрабатываем строки батча
        for row_idx in 0..batch.num_rows() {
            if !date_arr.is_valid(row_idx) {
                return Err(anyhow!("найдена null-дата в строке {}", row_idx));
            }

            let timestamp_micros = date_arr.value(row_idx);
            let date = DateTime::from_timestamp_micros(timestamp_micros)
                .map(|dt| dt.date_naive())
                .ok_or_else(|| {
                    anyhow!(
                        "timestamp {} выходит за допустимый диапазон дат",
                        timestamp_micros
                    )
                })?;

            match map.entry(date) {
                Entry::Vacant(entry) => {
                    let mut row_data = HashMap::with_capacity(indices.len());
                    for (name, col) in &float_cols {
                        let value = if col.is_valid(row_idx) {
                            Some(col.value(row_idx))
                        } else {
                            None
                        };
                        row_data.insert((*name).to_string(), value);
                    }
                    entry.insert(row_data);
                }
                Entry::Occupied(_) => {
                    return Err(anyhow!("найдена дублирующаяся дата {:?} в макро-файле", date));
                }
            }
        }
    }

    Ok(map)
}