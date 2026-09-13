//! Чтение баров из Parquet и утилиты для работы с тикерами.

use std::{
    fs::{self, File},
    io::ErrorKind,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result};
use arrow::array::{Array, Float64Array, Int64Array, UInt64Array};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use crate::config::OUTPUT_DIR;
use crate::data_loading::{desanitize_filename, sanitize_filename, BarAgg};

/// Загрузить бары для одного тикера из Parquet.
pub fn load_bars_for_ticker(symbol: &str) -> Result<Vec<BarAgg>> {
    if symbol.is_empty() {
        anyhow::bail!("Тикер не может быть пустым");
    }

    let safe_name = sanitize_filename(symbol);
    let file_path: PathBuf = Path::new(OUTPUT_DIR)
        .join("bars")
        .join(format!("{}.parquet", safe_name));

    let file = File::open(&file_path)
        .with_context(|| format!("Нет файла баров для тикера {}", symbol))?;

    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .with_context(|| format!("Не удалось создать читатель Parquet для {}", file_path.display()))?;
    let reader = builder.build()
        .with_context(|| format!("Не удалось инициализировать чтение Parquet из {}", file_path.display()))?;

    let mut bars = Vec::with_capacity(1024);

    for batch_result in reader {
        let batch = batch_result
            .with_context(|| format!("Ошибка чтения батча Parquet из {}", file_path.display()))?;

        let date_array = batch.column(0).as_any().downcast_ref::<Int64Array>()
            .context("Колонка 'date' должна быть Int64")?;
        let open_time_array = batch.column(1).as_any().downcast_ref::<Int64Array>()
            .context("Колонка 'open_time' должна быть Int64")?;
        let high_array = batch.column(2).as_any().downcast_ref::<Float64Array>()
            .context("Колонка 'high' должна быть Float64")?;
        let low_array = batch.column(3).as_any().downcast_ref::<Float64Array>()
            .context("Колонка 'low' должна быть Float64")?;
        let close_array = batch.column(4).as_any().downcast_ref::<Float64Array>()
            .context("Колонка 'close' должна быть Float64")?;
        let volume_array = batch.column(5).as_any().downcast_ref::<UInt64Array>()
            .context("Колонка 'volume' должна быть UInt64")?;
        let best_bid_array = batch.column(6).as_any().downcast_ref::<Float64Array>()
            .context("Колонка 'best_bid' должна быть Float64")?;
        let best_ask_array = batch.column(7).as_any().downcast_ref::<Float64Array>()
            .context("Колонка 'best_ask' должна быть Float64")?;

        for row_idx in 0..batch.num_rows() {
            let date = date_array.value(row_idx);
            let open_time = open_time_array.value(row_idx);
            let high = high_array.value(row_idx);
            let low = low_array.value(row_idx);
            let close = close_array.value(row_idx);
            let volume = volume_array.value(row_idx);
            let best_bid = if best_bid_array.is_null(row_idx) { None } else { Some(best_bid_array.value(row_idx)) };
            let best_ask = if best_ask_array.is_null(row_idx) { None } else { Some(best_ask_array.value(row_idx)) };

            // Валидация
            if date <= 0 { continue; }
            if open_time < 0 { continue; }
            if !high.is_finite() || !low.is_finite() || !close.is_finite() { continue; }
            if high < low { continue; }
            if close < low || close > high { continue; }
            if let (Some(bid), Some(ask)) = (best_bid, best_ask) {
                if bid > ask { continue; }
            }

            bars.push(BarAgg { date, open_time, high, low, close, volume, best_bid, best_ask });
        }
    }

    bars.sort_by_key(|b| (b.date, b.open_time));
    Ok(bars)
}

/// Получить список всех оригинальных тикеров, для которых есть файлы баров (Parquet).
pub fn list_available_tickers() -> Result<Vec<String>> {
    let bars_dir = Path::new(OUTPUT_DIR).join("bars");
    let mut tickers = Vec::new();

    let entries = match fs::read_dir(&bars_dir) {
        Ok(entries) => entries,
        Err(e) if e.kind() == ErrorKind::NotFound => return Ok(tickers),
        Err(e) => return Err(e).with_context(|| format!("Не удалось прочитать каталог {}", bars_dir.display())),
    };

    for entry in entries {
        let entry = match entry {
            Ok(entry) => entry,
            Err(_) => continue,
        };
        let path = entry.path();
        if !path.is_file() { continue; }
        if path.extension().and_then(|e| e.to_str()).map(|e| e.eq_ignore_ascii_case("parquet")).unwrap_or(false) {
            if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                let ticker = desanitize_filename(stem);
                if !ticker.is_empty() && !ticker.contains('/') && !ticker.contains('\\') {
                    tickers.push(ticker);
                }
            }
        }
    }

    tickers.sort();
    tickers.dedup();
    Ok(tickers)
}