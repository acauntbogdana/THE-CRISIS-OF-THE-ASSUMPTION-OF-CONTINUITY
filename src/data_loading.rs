use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fmt::Write as _,
    fs::{self, File, OpenOptions},
    path::{Path, PathBuf},
    sync::{Arc, OnceLock},
};

use anyhow::{anyhow, Context, Result};
use arrow::array::{
    Array, BooleanArray, Float64Array, Int64Array, UInt32Array, UInt64Array, StringArray,
};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use chrono::{Datelike, NaiveDate, TimeZone};
use chrono_tz::America::New_York;
use fs2::FileExt;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::properties::WriterProperties;
use serde::{Deserialize, Serialize};

use crate::config::{BAR_INTERVAL_MIN, DEEP_DIR, OUTPUT_DIR};

// Зарезервированные имена устройств Windows, которые нельзя использовать как имя файла.
const RESERVED_NAMES: &[&str] = &[
    "CON", "PRN", "AUX", "NUL",
    "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
    "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
];

#[derive(Serialize, Deserialize, Default)]
pub struct ProcessingState {
    pub processed_files: HashSet<String>,
    pub current_bars: HashMap<String, BarState>,
    pub pending_bars: HashMap<String, Vec<BarAgg>>,
    #[serde(skip)]
    pub written_bar_keys: HashMap<String, HashSet<(i64, i64)>>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct BarState {
    pub current_bar: Option<BarAgg>,
    pub bid_levels: BTreeMap<u64, u32>,
    pub ask_levels: BTreeMap<u64, u32>,
    pub trading_halted: bool,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct BarAgg {
    pub date: i64,
    pub open_time: i64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: u64,
    pub best_bid: Option<f64>,
    pub best_ask: Option<f64>,
}

impl BarAgg {
    pub fn new(date: NaiveDate, bar_start: i64, price: f64, volume: u64, bid: Option<f64>, ask: Option<f64>) -> Self {
        Self {
            date: date_to_unix_day(date),
            open_time: bar_start,
            high: price,
            low: price,
            close: price,
            volume,
            best_bid: bid,
            best_ask: ask,
        }
    }

    pub fn date_naive(&self) -> NaiveDate {
        unix_day_to_date(self.date)
    }
}

impl BarState {
    pub fn new() -> Self {
        Self {
            current_bar: None,
            bid_levels: BTreeMap::new(),
            ask_levels: BTreeMap::new(),
            trading_halted: false,
        }
    }

    fn best_bid(&self) -> Option<f64> {
        self.bid_levels.iter().next_back().map(|(k, _)| f64::from_bits(*k))
    }

    fn best_ask(&self) -> Option<f64> {
        self.ask_levels.iter().next().map(|(k, _)| f64::from_bits(*k))
    }
}

fn unix_epoch() -> NaiveDate {
    static EPOCH: OnceLock<NaiveDate> = OnceLock::new();
    *EPOCH.get_or_init(|| NaiveDate::from_ymd_opt(1970, 1, 1).expect("Неверная эпохальная дата"))
}

pub fn date_to_unix_day(date: NaiveDate) -> i64 {
    date.signed_duration_since(unix_epoch()).num_days()
}

pub fn unix_day_to_date(day: i64) -> NaiveDate {
    unix_epoch() + chrono::Duration::days(day)
}

pub fn list_parquet_files(dir: &str) -> Result<Vec<PathBuf>> {
    let mut files_with_dates: Vec<(NaiveDate, PathBuf)> = Vec::new();

    let entries = fs::read_dir(dir)
        .with_context(|| format!("Не удалось прочитать директорию {}", dir))?;

    for entry in entries {
        let entry = match entry {
            Ok(e) => e,
            Err(e) => {
                eprintln!("Предупреждение: ошибка при обходе директории {}: {}", dir, e);
                continue;
            }
        };
        let path = entry.path();
        if path.extension().map(|e| e.eq_ignore_ascii_case("parquet")).unwrap_or(false) {
            if let Ok(date) = date_from_filename(&path) {
                files_with_dates.push((date, path));
            } else {
                eprintln!("Предупреждение: файл с нераспознаваемым именем пропущен: {}", path.display());
            }
        }
    }

    files_with_dates.sort_by_key(|(date, _)| *date);
    Ok(files_with_dates.into_iter().map(|(_, path)| path).collect())
}

pub fn date_from_filename(path: &Path) -> Result<NaiveDate> {
    let fname = path
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or_else(|| anyhow!("Имя файла не является валидным UTF-8: {:?}", path))?;
    NaiveDate::parse_from_str(fname, "%Y-%m-%d")
        .with_context(|| format!("Неверный формат имени файла: {}", fname))
}

/// Санитизация имени файла с защитой от зарезервированных имён Windows.
pub fn sanitize_filename(name: &str) -> String {
    // Если имя совпадает с зарезервированным, добавляем префикс '_'
    if RESERVED_NAMES.iter().any(|res| name.eq_ignore_ascii_case(res)) {
        return format!("_{}", name);
    }

    let mut result = String::with_capacity(name.len() * 2);
    for ch in name.chars() {
        match ch {
            'a'..='z' | 'A'..='Z' | '0'..='9' => result.push(ch),
            '_' => result.push_str("__"),
            c if c.is_ascii() => {
                result.push('_');
                write!(&mut result, "{:02X}", c as u8).unwrap();
            }
            c => {
                write!(&mut result, "_u{:04X}", c as u32).unwrap();
            }
        }
    }
    result
}

/// Обратное преобразование с учётом префикса для зарезервированных имён.
pub fn desanitize_filename(sanitized: &str) -> String {
    // Если имя начинается с '_' и остаток является зарезервированным, убираем префикс
    if let Some(rest) = sanitized.strip_prefix('_') {
        if RESERVED_NAMES.iter().any(|res| rest.eq_ignore_ascii_case(res)) {
            return rest.to_string();
        }
    }

    let mut result = String::with_capacity(sanitized.len());
    let bytes = sanitized.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'_' {
            if i + 1 < bytes.len() {
                if bytes[i + 1] == b'_' {
                    result.push('_');
                    i += 2;
                    continue;
                } else if i + 2 < bytes.len() && bytes[i + 1].is_ascii_hexdigit() && bytes[i + 2].is_ascii_hexdigit() {
                    if let Ok(byte) = u8::from_str_radix(&sanitized[i + 1..i + 3], 16) {
                        if let Some(ch) = char::from_u32(byte as u32) {
                            result.push(ch);
                            i += 3;
                            continue;
                        }
                    }
                } else if i + 1 < bytes.len() && bytes[i + 1] == b'u' {
                    if i + 5 < bytes.len() && bytes[i + 2..i + 6].iter().all(|b| b.is_ascii_hexdigit()) {
                        if let Ok(code) = u32::from_str_radix(&sanitized[i + 2..i + 6], 16) {
                            if let Some(ch) = char::from_u32(code) {
                                result.push(ch);
                                i += 6;
                                continue;
                            }
                        }
                    }
                }
            }
            result.push('_');
            i += 1;
        } else {
            let ch = sanitized[i..].chars().next().unwrap();
            result.push(ch);
            i += ch.len_utf8();
        }
    }
    result
}

#[derive(Default, Debug)]
pub struct ProcessStats {
    skipped_null: usize,
    skipped_ts_before_midnight: usize,
    skipped_ts_after_day: usize,
    skipped_halted: usize,
    skipped_trade_break: usize,
    skipped_invalid_price: usize,
    skipped_zero_size: usize,
    skipped_missing_columns: usize,
    skipped_unknown_side: usize,
}

impl ProcessStats {
    pub fn print(&self, filename: &str) {
        eprintln!(
            "Статистика обработки {}: null={}, ts_early={}, ts_late={}, halted={}, trade_break={}, invalid_price={}, zero_size={}, missing_cols={}, unknown_side={}",
            filename,
            self.skipped_null,
            self.skipped_ts_before_midnight,
            self.skipped_ts_after_day,
            self.skipped_halted,
            self.skipped_trade_break,
            self.skipped_invalid_price,
            self.skipped_zero_size,
            self.skipped_missing_columns,
            self.skipped_unknown_side,
        );
    }
}

pub fn process_deep_file(path: &Path, date: NaiveDate, state: &mut ProcessingState) -> Result<ProcessStats> {
    let file = File::open(path)
        .with_context(|| format!("Не удалось открыть файл {:?}", path))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let schema = builder.schema().clone();
    let reader = builder.with_batch_size(100_000).build()?;

    let idx_msg_type = schema.index_of("msg_type").map_err(|_| anyhow!("Отсутствует столбец msg_type"))?;
    let idx_timestamp = schema.index_of("timestamp").map_err(|_| anyhow!("Отсутствует timestamp"))?;
    let idx_symbol = schema.index_of("symbol").map_err(|_| anyhow!("Отсутствует symbol"))?;

    let idx_price = schema.index_of("price").ok();
    let idx_size = schema.index_of("size").ok();
    let idx_side = schema.index_of("side").ok();
    let idx_trading_status = schema.index_of("trading_status").ok();
    let idx_system_event = schema.index_of("system_event").ok();
    let idx_is_trade_break = schema.index_of("is_trade_break").ok();

    let et_midnight = New_York
        .with_ymd_and_hms(date.year(), date.month(), date.day(), 0, 0, 0)
        .single()
        .ok_or_else(|| anyhow!("Невозможно создать полночь в NY для даты {}", date))?;
    let et_midnight_utc_ns = et_midnight.timestamp_nanos_opt()
        .ok_or_else(|| anyhow!("Переполнение при конвертации времени"))?;

    let session_start = New_York
        .with_ymd_and_hms(date.year(), date.month(), date.day(), 9, 30, 0)
        .single()
        .ok_or_else(|| anyhow!("Невозможно создать время начала сессии"))?;
    let session_start_utc_ns = session_start.timestamp_nanos_opt()
        .ok_or_else(|| anyhow!("Переполнение при конвертации времени"))?;

    let session_end = New_York
        .with_ymd_and_hms(date.year(), date.month(), date.day(), 16, 0, 0)
        .single()
        .ok_or_else(|| anyhow!("Невозможно создать время конца сессии"))?;
    let session_end_utc_ns = session_end.timestamp_nanos_opt()
        .ok_or_else(|| anyhow!("Переполнение при конвертации времени"))?;

    let day_ns = 86_400i64 * 1_000_000_000;
    let unix_day = date_to_unix_day(date);

    let mut stats = ProcessStats::default();

    for batch in reader {
        let batch = batch?;
        let msg_type = batch.column(idx_msg_type).as_any().downcast_ref::<StringArray>()
            .ok_or_else(|| anyhow!("msg_type не StringArray"))?;
        let timestamp = batch.column(idx_timestamp).as_any().downcast_ref::<Int64Array>()
            .ok_or_else(|| anyhow!("timestamp не Int64Array"))?;
        let symbol = batch.column(idx_symbol).as_any().downcast_ref::<StringArray>()
            .ok_or_else(|| anyhow!("symbol не StringArray"))?;

        let price_arr = idx_price.map(|idx| {
            batch.column(idx).as_any().downcast_ref::<Float64Array>()
                .ok_or_else(|| anyhow!("price не Float64Array"))
        }).transpose()?;

        let size_arr = idx_size.map(|idx| {
            batch.column(idx).as_any().downcast_ref::<UInt32Array>()
                .ok_or_else(|| anyhow!("size не UInt32Array"))
        }).transpose()?;

        let _side_arr = idx_side.map(|idx| {
            batch.column(idx).as_any().downcast_ref::<StringArray>()
                .ok_or_else(|| anyhow!("side не StringArray"))
        }).transpose()?;

        let trading_status_arr = idx_trading_status.map(|idx| {
            batch.column(idx).as_any().downcast_ref::<StringArray>()
                .ok_or_else(|| anyhow!("trading_status не StringArray"))
        }).transpose()?;

        let system_event_arr = idx_system_event.map(|idx| {
            batch.column(idx).as_any().downcast_ref::<StringArray>()
                .ok_or_else(|| anyhow!("system_event не StringArray"))
        }).transpose()?;

        let is_trade_break_arr = idx_is_trade_break.map(|idx| {
            batch.column(idx).as_any().downcast_ref::<BooleanArray>()
                .ok_or_else(|| anyhow!("is_trade_break не BooleanArray"))
        }).transpose()?;

        for i in 0..batch.num_rows() {
            if msg_type.is_null(i) || symbol.is_null(i) || timestamp.is_null(i) {
                stats.skipped_null += 1;
                continue;
            }
            let sym = symbol.value(i);
            let ts = timestamp.value(i);

            if ts < session_start_utc_ns {
                stats.skipped_ts_before_midnight += 1;
                continue;
            }
            if ts >= session_end_utc_ns {
                stats.skipped_ts_after_day += 1;
                continue;
            }

            let ts_from_et_midnight = ts - et_midnight_utc_ns;

            let bar_state = state.current_bars
                .entry(sym.to_string())
                .or_insert_with(BarState::new);

            match msg_type.value(i) {
                "trading_status" => {
                    if let Some(arr) = &trading_status_arr {
                        if !arr.is_null(i) {
                            bar_state.trading_halted = arr.value(i) == "halted";
                        }
                    }
                }
                "system_event" => {
                    if let Some(arr) = &system_event_arr {
                        if !arr.is_null(i) {
                            let event = arr.value(i);
                            match event {
                                "start_of_messages" => {
                                    bar_state.bid_levels.clear();
                                    bar_state.ask_levels.clear();
                                    bar_state.trading_halted = false;
                                }
                                "end_of_messages" => {
                                    if let Some(prev_bar) = bar_state.current_bar.take() {
                                        state.pending_bars.entry(sym.to_string()).or_default().push(prev_bar);
                                    }
                                    bar_state.bid_levels.clear();
                                    bar_state.ask_levels.clear();
                                }
                                _ => {}
                            }
                        }
                    }
                }
                "price_level_update_buy" => {
                    let (Some(price), Some(size)) = (
                        price_arr.as_ref(),
                        size_arr.as_ref(),
                    ) else {
                        stats.skipped_missing_columns += 1;
                        continue;
                    };
                    if price.is_null(i) || size.is_null(i) {
                        stats.skipped_null += 1;
                        continue;
                    }
                    let px = price.value(i);
                    let sz = size.value(i);
                    if !px.is_finite() || px <= 0.0 {
                        stats.skipped_invalid_price += 1;
                        continue;
                    }
                    let price_bits = px.to_bits();
                    if sz > 0 {
                        bar_state.bid_levels.insert(price_bits, sz);
                    } else {
                        bar_state.bid_levels.remove(&price_bits);
                    }
                }
                "price_level_update_sell" => {
                    let (Some(price), Some(size)) = (
                        price_arr.as_ref(),
                        size_arr.as_ref(),
                    ) else {
                        stats.skipped_missing_columns += 1;
                        continue;
                    };
                    if price.is_null(i) || size.is_null(i) {
                        stats.skipped_null += 1;
                        continue;
                    }
                    let px = price.value(i);
                    let sz = size.value(i);
                    if !px.is_finite() || px <= 0.0 {
                        stats.skipped_invalid_price += 1;
                        continue;
                    }
                    let price_bits = px.to_bits();
                    if sz > 0 {
                        bar_state.ask_levels.insert(price_bits, sz);
                    } else {
                        bar_state.ask_levels.remove(&price_bits);
                    }
                }
                "trade_report" => {
                    if bar_state.trading_halted {
                        stats.skipped_halted += 1;
                        continue;
                    }
                    let (Some(price), Some(size)) = (price_arr.as_ref(), size_arr.as_ref()) else {
                        stats.skipped_missing_columns += 1;
                        continue;
                    };
                    if price.is_null(i) || size.is_null(i) {
                        stats.skipped_null += 1;
                        continue;
                    }
                    if let Some(arr) = &is_trade_break_arr {
                        if !arr.is_null(i) && arr.value(i) {
                            stats.skipped_trade_break += 1;
                            continue;
                        }
                    }
                    let px = price.value(i);
                    let sz = size.value(i) as u64;
                    if !px.is_finite() || px <= 0.0 {
                        stats.skipped_invalid_price += 1;
                        continue;
                    }
                    if sz == 0 {
                        stats.skipped_zero_size += 1;
                        continue;
                    }

                    let bar_interval_ns = BAR_INTERVAL_MIN as i64 * 60 * 1_000_000_000;
                    let bar_start = (ts_from_et_midnight / bar_interval_ns) * bar_interval_ns;
                    if bar_start >= day_ns {
                        stats.skipped_ts_after_day += 1;
                        continue;
                    }

                    let bid = bar_state.best_bid();
                    let ask = bar_state.best_ask();

                    // Мягкая очистка: если bid > ask, очищаем котировки
                    let (bid_final, ask_final) = match (bid, ask) {
                        (Some(b), Some(a)) if b <= a => (Some(b), Some(a)),
                        _ => (None, None),
                    };

                    match &mut bar_state.current_bar {
                        Some(agg) if agg.open_time == bar_start && agg.date == unix_day => {
                            agg.high = agg.high.max(px);
                            agg.low = agg.low.min(px);
                            agg.close = px;
                            agg.volume += sz;
                            agg.best_bid = bid_final;
                            agg.best_ask = ask_final;
                        }
                        _ => {
                            if let Some(prev) = bar_state.current_bar.take() {
                                state.pending_bars.entry(sym.to_string()).or_default().push(prev);
                            }
                            bar_state.current_bar = Some(BarAgg::new(
                                date,
                                bar_start,
                                px,
                                sz,
                                bid_final,
                                ask_final,
                            ));
                        }
                    }
                }
                _ => {}
            }
        }
    }

    Ok(stats)
}

fn write_bars_to_parquet(file_path: &Path, bars: &[&BarAgg]) -> Result<()> {
    if bars.is_empty() {
        return Ok(());
    }

    let mut date = Vec::with_capacity(bars.len());
    let mut open_time = Vec::with_capacity(bars.len());
    let mut high = Vec::with_capacity(bars.len());
    let mut low = Vec::with_capacity(bars.len());
    let mut close = Vec::with_capacity(bars.len());
    let mut volume = Vec::with_capacity(bars.len());
    let mut best_bid = Vec::with_capacity(bars.len());
    let mut best_ask = Vec::with_capacity(bars.len());

    for bar in bars {
        date.push(bar.date);
        open_time.push(bar.open_time);
        high.push(bar.high);
        low.push(bar.low);
        close.push(bar.close);
        volume.push(bar.volume);
        best_bid.push(bar.best_bid);
        best_ask.push(bar.best_ask);
    }

    let fields = vec![
        Field::new("date", DataType::Int64, false),
        Field::new("open_time", DataType::Int64, false),
        Field::new("high", DataType::Float64, false),
        Field::new("low", DataType::Float64, false),
        Field::new("close", DataType::Float64, false),
        Field::new("volume", DataType::UInt64, false),
        Field::new("best_bid", DataType::Float64, true),
        Field::new("best_ask", DataType::Float64, true),
    ];
    let schema = Arc::new(Schema::new(fields));

    let date_array = Int64Array::from(date);
    let open_time_array = Int64Array::from(open_time);
    let high_array = Float64Array::from(high);
    let low_array = Float64Array::from(low);
    let close_array = Float64Array::from(close);
    let volume_array = UInt64Array::from(volume);
    let best_bid_array = Float64Array::from(best_bid);
    let best_ask_array = Float64Array::from(best_ask);

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(date_array),
            Arc::new(open_time_array),
            Arc::new(high_array),
            Arc::new(low_array),
            Arc::new(close_array),
            Arc::new(volume_array),
            Arc::new(best_bid_array),
            Arc::new(best_ask_array),
        ],
    )?;

    let file = File::create(file_path)?;
    let props = WriterProperties::builder()
        .set_compression(Compression::ZSTD(ZstdLevel::default()))
        .build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props))?;
    writer.write(&batch)?;
    writer.close()?;

    Ok(())
}

fn read_bars_from_parquet(path: &Path) -> Result<Vec<BarAgg>> {
    let file = File::open(path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let reader = builder.build()?;
    let mut bars = Vec::new();

    for batch in reader {
        let batch = batch?;
        let date = batch.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
        let open_time = batch.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
        let high = batch.column(2).as_any().downcast_ref::<Float64Array>().unwrap();
        let low = batch.column(3).as_any().downcast_ref::<Float64Array>().unwrap();
        let close = batch.column(4).as_any().downcast_ref::<Float64Array>().unwrap();
        let volume = batch.column(5).as_any().downcast_ref::<UInt64Array>().unwrap();
        let best_bid = batch.column(6).as_any().downcast_ref::<Float64Array>().unwrap();
        let best_ask = batch.column(7).as_any().downcast_ref::<Float64Array>().unwrap();

        for i in 0..batch.num_rows() {
            bars.push(BarAgg {
                date: date.value(i),
                open_time: open_time.value(i),
                high: high.value(i),
                low: low.value(i),
                close: close.value(i),
                volume: volume.value(i),
                best_bid: if best_bid.is_valid(i) { Some(best_bid.value(i)) } else { None },
                best_ask: if best_ask.is_valid(i) { Some(best_ask.value(i)) } else { None },
            });
        }
    }

    Ok(bars)
}

fn load_written_keys(bars_dir: &Path) -> Result<HashMap<String, HashSet<(i64, i64)>>> {
    let mut map = HashMap::new();
    if !bars_dir.exists() {
        return Ok(map);
    }
    for entry in fs::read_dir(bars_dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("parquet") {
            continue;
        }
        let bars = read_bars_from_parquet(&path)?;
        let mut keys = HashSet::new();
        for bar in &bars {
            keys.insert((bar.date, bar.open_time));
        }
        let sym = path.file_stem().and_then(|s| s.to_str()).unwrap_or("").to_string();
        if !sym.is_empty() {
            map.entry(sym).or_insert_with(HashSet::new).extend(keys);
        }
    }
    Ok(map)
}

pub fn flush_pending_bars(state: &mut ProcessingState, output_dir: &str) -> Result<()> {
    let bars_dir = Path::new(output_dir).join("bars");
    fs::create_dir_all(&bars_dir)?;

    for (sym, bars) in state.pending_bars.iter_mut() {
        if bars.is_empty() {
            continue;
        }
        let safe_sym = sanitize_filename(sym);
        let file_path = bars_dir.join(format!("{}.parquet", safe_sym));
        let temp_path = bars_dir.join(format!("{}.parquet.tmp", safe_sym));

        let written_keys = state.written_bar_keys
            .entry(safe_sym.clone())
            .or_default();

        let new_bars: Vec<&BarAgg> = bars.iter()
            .filter(|bar| !written_keys.contains(&(bar.date, bar.open_time)))
            .collect();
        if new_bars.is_empty() {
            continue;
        }

        let mut existing_bars: Vec<BarAgg> = Vec::new();
        if file_path.exists() {
            existing_bars = read_bars_from_parquet(&file_path)?;
        }

        let mut all_bars: Vec<&BarAgg> = existing_bars.iter().collect();
        all_bars.extend(new_bars.iter().copied());

        write_bars_to_parquet(&temp_path, &all_bars)?;
        fs::rename(&temp_path, &file_path)?;

        for bar in new_bars {
            written_keys.insert((bar.date, bar.open_time));
        }
    }

    state.pending_bars.clear();
    Ok(())
}

fn create_lock_file(lock_path: &str) -> Result<File> {
    let file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(false)
        .open(lock_path)
        .with_context(|| format!("Не удалось открыть lock-файл {}", lock_path))?;

    file.try_lock_exclusive()
        .map_err(|e| anyhow!("Не удалось заблокировать lock-файл {}: {}", lock_path, e))?;
    Ok(file)
}

pub fn build_all_bars() -> Result<()> {
    fs::create_dir_all(OUTPUT_DIR)?;
    let state_file = format!("{}/state.json", OUTPUT_DIR);

    let lock_file_path = format!("{}/build.lock", OUTPUT_DIR);
    let _lock_file = create_lock_file(&lock_file_path)?;

    let mut state: ProcessingState = if Path::new(&state_file).exists() {
        match fs::read_to_string(&state_file) {
            Ok(data) => match serde_json::from_str(&data) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("Предупреждение: не удалось разобрать state.json ({}), начинаем с пустого состояния", e);
                    ProcessingState::default()
                }
            },
            Err(e) => {
                eprintln!("Предупреждение: не удалось прочитать state.json ({}), начинаем с пустого состояния", e);
                ProcessingState::default()
            }
        }
    } else {
        ProcessingState::default()
    };

    let bars_dir = Path::new(OUTPUT_DIR).join("bars");
    if bars_dir.exists() {
        state.written_bar_keys = load_written_keys(&bars_dir)?;
    } else {
        state.written_bar_keys.clear();
    }

    if !state.pending_bars.is_empty() {
        eprintln!("Обнаружены незаписанные бары из предыдущего запуска, выполняется запись...");
        if let Err(e) = flush_pending_bars(&mut state, OUTPUT_DIR) {
            eprintln!("Ошибка записи незаписанных баров: {}", e);
            let state_json = serde_json::to_string_pretty(&state)?;
            if let Err(se) = fs::write(&state_file, state_json) {
                eprintln!("Ошибка сохранения state.json: {}", se);
            }
            return Err(e);
        }
        let state_json = serde_json::to_string_pretty(&state)?;
        if let Err(e) = fs::write(&state_file, state_json) {
            eprintln!("Ошибка сохранения state.json: {}", e);
        }
    }

    let files = list_parquet_files(DEEP_DIR)?;
    for (i, path) in files.iter().enumerate() {
        let fname = path.file_name()
            .and_then(|s| s.to_str())
            .ok_or_else(|| anyhow!("Файл без валидного имени: {:?}", path))?
            .to_string();
        if state.processed_files.contains(&fname) {
            continue;
        }
        let date = date_from_filename(path)?;
        println!("Обработка {}/{}: {} ({})", i + 1, files.len(), fname, date);

        let stats = match process_deep_file(path, date, &mut state) {
            Ok(stats) => stats,
            Err(e) => {
                eprintln!("Ошибка обработки файла {}: {}", path.display(), e);
                let state_json = serde_json::to_string_pretty(&state)?;
                if let Err(se) = fs::write(&state_file, state_json) {
                    eprintln!("Ошибка сохранения state.json: {}", se);
                }
                return Err(e);
            }
        };
        stats.print(&fname);

        if let Err(e) = flush_pending_bars(&mut state, OUTPUT_DIR) {
            eprintln!("Ошибка записи баров: {}", e);
            let state_json = serde_json::to_string_pretty(&state)?;
            if let Err(se) = fs::write(&state_file, state_json) {
                eprintln!("Ошибка сохранения state.json: {}", se);
            }
            return Err(e);
        }

        state.processed_files.insert(fname.clone());
        let state_json = serde_json::to_string_pretty(&state)?;
        if let Err(e) = fs::write(&state_file, state_json) {
            eprintln!("Ошибка сохранения state.json: {}", e);
        }
    }

    for (sym, bar_state) in state.current_bars.iter_mut() {
        if let Some(bar) = bar_state.current_bar.take() {
            state.pending_bars.entry(sym.clone()).or_default().push(bar);
        }
    }
    if let Err(e) = flush_pending_bars(&mut state, OUTPUT_DIR) {
        eprintln!("Ошибка записи финальных баров: {}", e);
        let state_json = serde_json::to_string_pretty(&state)?;
        if let Err(se) = fs::write(&state_file, state_json) {
            eprintln!("Ошибка сохранения state.json: {}", se);
        }
        return Err(e);
    }

    let state_json = serde_json::to_string_pretty(&state)?;
    if let Err(e) = fs::write(&state_file, state_json) {
        eprintln!("Ошибка сохранения state.json: {}", e);
    }

    Ok(())
}