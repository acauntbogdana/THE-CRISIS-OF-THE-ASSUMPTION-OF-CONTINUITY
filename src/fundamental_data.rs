// Загрузка и обработка фундаментальных данных SEC.
//
// Логика:
//   - балансовые (qtrs=0) теги берутся как мгновенные снимки;
//   - доходные теги накапливаются отдельно:
//       * qtrs=1 — квартальные данные, из которых строится TTM (сумма 4 кварталов);
//       * qtrs=4 — годовые данные, используются как fallback, если TTM построить нельзя;
//   - при вычислении показателей балансовая и доходная запись берутся независимо
//     (последняя доступная для каждой).

use std::{
    collections::{BTreeMap, HashMap},
    fs::File,
    path::PathBuf,
};

use anyhow::{anyhow, Result};
use arrow::array::{
    Array, Date32Array, Date64Array, Float64Array, Int64Array, LargeStringArray,
    TimestampMicrosecondArray, TimestampNanosecondArray, StringArray,
};
use chrono::{DateTime, Datelike, NaiveDate};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use crate::config::FUND_DIR;

// ====================== Теги ======================

/// Возвращает числовой идентификатор тега с учётом синонимов.
#[inline]
fn tag_id(tag: &str) -> Option<u8> {
    Some(match tag {
        // --- Балансовые (0..14) ---
        "Assets" => 0,
        "Liabilities" => 1,
        "StockholdersEquity"
        | "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest" => 2,
        "CashAndCashEquivalents"
        | "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents" => 3,
        "AssetsCurrent" => 4,
        "LiabilitiesCurrent" => 5,
        "InventoryNet" => 6,
        "AccountsReceivableNetCurrent" => 7,
        "AccountsPayableCurrent" => 8,
        "SharesOutstanding" | "CommonStockSharesOutstanding" => 9,
        "LongTermDebtNoncurrent"
        | "LongTermDebt"
        | "LongTermDebtAndCapitalLeaseObligations" => 10,
        "ShortTermBorrowings" | "ShortTermDebt" => 11,
        "Debt" | "DebtCurrent" | "LongTermDebtCurrent" => 12,
        "RetainedEarningsAccumulatedDeficit" => 13,

        // --- Доходные (15..27) ---
        "Revenues"
        | "RevenueFromContractWithCustomerExcludingAssessedTax"
        | "RevenueFromContractWithCustomerIncludingAssessedTax"
        | "SalesRevenueNet"
        | "SalesRevenueGoodsNet"
        | "SalesRevenueServicesNet" => 15,

        "NetIncomeLoss"
        | "NetIncomeLossAvailableToCommonStockholdersBasic"
        | "ProfitLoss" => 16,

        "OperatingCashFlow"
        | "NetCashProvidedByUsedInOperatingActivities" => 17,

        "OperatingIncomeLoss" => 18,

        "GrossProfit" => 19,

        "CostOfRevenue"
        | "CostOfGoodsAndServicesSold"
        | "CostOfGoodsSold"
        | "CostOfServices" => 20,

        "InterestExpense" | "InterestExpenseDebt" => 21,
        "IncomeTaxExpenseBenefit" => 22,
        "DepreciationDepletionAndAmortization"
        | "DepreciationAmortizationAndAccretionNet"
        | "DepreciationAndAmortization" => 23,
        "ResearchAndDevelopmentExpense" => 24,
        "SellingGeneralAndAdministrativeExpense"
        | "GeneralAndAdministrativeExpense" => 25,
        "EarningsPerShareBasic" => 26,
        "EarningsPerShareDiluted" => 27,

        _ => return None,
    })
}

// ====================== Вспомогательные ======================

#[inline]
fn parse_yyyymmdd(val: i64) -> Option<NaiveDate> {
    let y = (val / 10000) as i32;
    let m = ((val / 100) % 100) as u32;
    let d = (val % 100) as u32;
    NaiveDate::from_ymd_opt(y, m, d)
}

#[inline]
fn timestamp_micros_to_date(ts: i64) -> Option<NaiveDate> {
    DateTime::from_timestamp_micros(ts).map(|dt| dt.date_naive())
}

#[inline]
fn date32_to_date(days: i32) -> Option<NaiveDate> {
    chrono::NaiveDate::from_num_days_from_ce_opt((days as i64 + 719_163) as i32)
}

#[inline]
fn date64_to_date(millis: i64) -> Option<NaiveDate> {
    DateTime::from_timestamp_millis(millis).map(|dt| dt.date_naive())
}

#[inline]
fn date_to_yyyymmdd(date: NaiveDate) -> i64 {
    date.year() as i64 * 10000 + date.month() as i64 * 100 + date.day() as i64
}

// ====================== Структуры ======================

#[derive(Clone, Debug)]
pub struct FundamentalRecord {
    pub date: NaiveDate,
    pub filed: NaiveDate,

    pub assets: f64,
    pub liabilities: f64,
    pub equity: f64,
    pub cash: f64,
    pub assets_current: f64,
    pub liabilities_current: f64,
    pub inventory: f64,
    pub receivables: f64,
    pub payables: f64,
    pub shares_outstanding: f64,
    pub long_term_debt: f64,
    pub short_term_debt: f64,
    pub total_debt: f64,
    pub retained_earnings: f64,
    pub shares_common: f64,

    pub revenue: f64,
    pub net_income: f64,
    pub operating_cash_flow: f64,
    pub operating_income: f64,
    pub gross_profit: f64,
    pub cost_of_revenue: f64,
    pub interest_expense: f64,
    pub income_tax: f64,
    pub depreciation_amortization: f64,
    pub rd_expense: f64,
    pub sga_expense: f64,
    pub eps_basic: f64,
    pub eps_diluted: f64,
}

impl Default for FundamentalRecord {
    fn default() -> Self {
        Self {
            date: NaiveDate::from_ymd_opt(1970, 1, 1).unwrap(),
            filed: NaiveDate::from_ymd_opt(1970, 1, 1).unwrap(),
            assets: f64::NAN,
            liabilities: f64::NAN,
            equity: f64::NAN,
            cash: f64::NAN,
            assets_current: f64::NAN,
            liabilities_current: f64::NAN,
            inventory: f64::NAN,
            receivables: f64::NAN,
            payables: f64::NAN,
            shares_outstanding: f64::NAN,
            long_term_debt: f64::NAN,
            short_term_debt: f64::NAN,
            total_debt: f64::NAN,
            retained_earnings: f64::NAN,
            shares_common: f64::NAN,
            revenue: f64::NAN,
            net_income: f64::NAN,
            operating_cash_flow: f64::NAN,
            operating_income: f64::NAN,
            gross_profit: f64::NAN,
            cost_of_revenue: f64::NAN,
            interest_expense: f64::NAN,
            income_tax: f64::NAN,
            depreciation_amortization: f64::NAN,
            rd_expense: f64::NAN,
            sga_expense: f64::NAN,
            eps_basic: f64::NAN,
            eps_diluted: f64::NAN,
        }
    }
}

#[derive(Debug, Clone, Default)]
struct QuarterlyIncome {
    date: NaiveDate,
    filed: NaiveDate,
    revenue: f64,
    net_income: f64,
    operating_cash_flow: f64,
    operating_income: f64,
    gross_profit: f64,
    cost_of_revenue: f64,
    interest_expense: f64,
    income_tax: f64,
    depreciation_amortization: f64,
    rd_expense: f64,
    sga_expense: f64,
    eps_basic: f64,
    eps_diluted: f64,
}

#[derive(Debug, Clone)]
pub struct FundamentalFeatures {
    pub roa: f64,
    pub debt_to_equity: f64,
    pub cash_ratio: f64,
    pub quick_ratio: f64,
    pub gross_margin: f64,
    pub operating_margin: f64,
    pub net_margin: f64,
    pub interest_coverage: f64,
    pub asset_turnover: f64,
    pub rd_intensity: f64,
    pub sga_intensity: f64,
    pub eps_growth: f64,
    pub roe: f64,
    pub leverage: f64,
    pub revenue_growth: f64,
    pub net_income_growth: f64,
    pub current_ratio: f64,
}

pub const FUNDAMENTAL_FEATURE_COUNT: usize = 17;

// ====================== Загрузка ======================

pub fn load_all_fundamental_records(
    tickers_map: &BTreeMap<i64, String>,
) -> Result<HashMap<String, Vec<FundamentalRecord>>> {
    let files = list_parquet_files(FUND_DIR)?;

    // Балансовые записи (qtrs == 0)
    let mut balance_map: HashMap<(i64, NaiveDate), FundamentalRecord> = HashMap::new();
    // Квартальные доходы (qtrs == 1)
    let mut quarterly_map: HashMap<(i64, NaiveDate), QuarterlyIncome> = HashMap::new();
    // Годовые доходы (qtrs == 4) — fallback
    let mut annual_map: HashMap<(i64, NaiveDate), FundamentalRecord> = HashMap::new();

    // Дедупликация: для каждого (cik, date, tag_id) храним лучший filed
    let mut best_balance: HashMap<(i64, NaiveDate, u8), i64> = HashMap::new();
    let mut best_quarter: HashMap<(i64, NaiveDate, u8), i64> = HashMap::new();
    let mut best_annual: HashMap<(i64, NaiveDate, u8), i64> = HashMap::new();

    for path in files {
        let file = File::open(path)?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
        let schema = builder.schema().clone();

        let idx_cik = schema.index_of("cik").map_err(|_| anyhow!("колонка cik не найдена"))?;
        let idx_tag = schema.index_of("tag").map_err(|_| anyhow!("колонка tag не найдена"))?;
        let idx_value = schema.index_of("value").map_err(|_| anyhow!("колонка value не найдена"))?;
        let idx_ddate = schema.index_of("ddate").map_err(|_| anyhow!("колонка ddate не найдена"))?;
        let idx_qtrs = schema.index_of("qtrs").map_err(|_| anyhow!("колонка qtrs не найдена"))?;
        let idx_filed = schema.index_of("filed").map_err(|_| anyhow!("колонка filed не найдена"))?;

        let reader = builder.with_batch_size(100_000).build()?;
        for batch in reader {
            let batch = batch?;

            let cik_arr = batch.column(idx_cik).as_any().downcast_ref::<Int64Array>()
                .ok_or_else(|| anyhow!("cik не Int64Array"))?;
            let tag_large = batch.column(idx_tag).as_any().downcast_ref::<LargeStringArray>();
            let tag_utf8 = batch.column(idx_tag).as_any().downcast_ref::<StringArray>();

            let get_tag = |i: usize| -> Option<&str> {
                if let Some(arr) = tag_large {
                    if arr.is_valid(i) { return Some(arr.value(i)); }
                }
                if let Some(arr) = tag_utf8 {
                    if arr.is_valid(i) { return Some(arr.value(i)); }
                }
                None
            };

            let value_arr = batch.column(idx_value).as_any().downcast_ref::<Float64Array>()
                .ok_or_else(|| anyhow!("value не Float64Array"))?;
            let ddate_arr = batch.column(idx_ddate).as_any().downcast_ref::<TimestampMicrosecondArray>()
                .ok_or_else(|| anyhow!("ddate не TimestampMicrosecondArray"))?;
            let qtrs_arr = batch.column(idx_qtrs).as_any().downcast_ref::<Int64Array>()
                .ok_or_else(|| anyhow!("qtrs не Int64Array"))?;

            let filed_col = batch.column(idx_filed);
            let filed_int_arr = filed_col.as_any().downcast_ref::<Int64Array>();
            let filed_micro_arr = filed_col.as_any().downcast_ref::<TimestampMicrosecondArray>();
            let filed_nano_arr = filed_col.as_any().downcast_ref::<TimestampNanosecondArray>();
            let filed_date32_arr = filed_col.as_any().downcast_ref::<Date32Array>();
            let filed_date64_arr = filed_col.as_any().downcast_ref::<Date64Array>();

            for i in 0..batch.num_rows() {
                if !(cik_arr.is_valid(i) && value_arr.is_valid(i) && ddate_arr.is_valid(i) && qtrs_arr.is_valid(i)) {
                    continue;
                }

                let tag_str = match get_tag(i) { Some(t) => t, None => continue };
                let id = match tag_id(tag_str) { Some(id) => id, None => continue };
                let qtrs = qtrs_arr.value(i);

                let is_balance = id < 15 && qtrs == 0;
                let is_quarter = id >= 15 && qtrs == 1;
                let is_annual = id >= 15 && qtrs == 4;
                if !(is_balance || is_quarter || is_annual) {
                    continue;
                }

                let ddate_ts = ddate_arr.value(i);
                let date = match timestamp_micros_to_date(ddate_ts) {
                    Some(d) => d,
                    None => continue,
                };

                let filed_int = if let Some(arr) = filed_int_arr {
                    if arr.is_valid(i) { Some(arr.value(i)) } else { None }
                } else if let Some(arr) = filed_micro_arr {
                    if arr.is_valid(i) {
                        timestamp_micros_to_date(arr.value(i)).map(date_to_yyyymmdd)
                    } else { None }
                } else if let Some(arr) = filed_nano_arr {
                    if arr.is_valid(i) {
                        let dt = DateTime::from_timestamp_nanos(arr.value(i));
                        Some(date_to_yyyymmdd(dt.date_naive()))
                    } else { None }
                } else if let Some(arr) = filed_date32_arr {
                    if arr.is_valid(i) {
                        date32_to_date(arr.value(i)).map(date_to_yyyymmdd)
                    } else { None }
                } else if let Some(arr) = filed_date64_arr {
                    if arr.is_valid(i) {
                        date64_to_date(arr.value(i)).map(date_to_yyyymmdd)
                    } else { None }
                } else {
                    None
                };

                let filed_int = match filed_int { Some(v) => v, None => continue };
                let filed_date = match parse_yyyymmdd(filed_int) {
                    Some(d) => d,
                    None => continue,
                };
                let cik = cik_arr.value(i);
                let value = value_arr.value(i);

                if is_balance {
                    let key = (cik, date, id);
                    let better = match best_balance.get(&key) {
                        Some(&prev) => filed_int > prev,
                        None => true,
                    };
                    if !better { continue; }
                    best_balance.insert(key, filed_int);

                    let entry = balance_map.entry((cik, date))
                        .or_insert_with(|| FundamentalRecord { date, ..Default::default() });
                    if filed_date > entry.filed { entry.filed = filed_date; }
                    match id {
                        0 => entry.assets = value,
                        1 => entry.liabilities = value,
                        2 => entry.equity = value,
                        3 => entry.cash = value,
                        4 => entry.assets_current = value,
                        5 => entry.liabilities_current = value,
                        6 => entry.inventory = value,
                        7 => entry.receivables = value,
                        8 => entry.payables = value,
                        9 => entry.shares_outstanding = value,
                        10 => entry.long_term_debt = value,
                        11 => entry.short_term_debt = value,
                        12 => entry.total_debt = value,
                        13 => entry.retained_earnings = value,
                        14 => entry.shares_common = value,
                        _ => {}
                    }
                } else if is_quarter {
                    let key = (cik, date, id);
                    let better = match best_quarter.get(&key) {
                        Some(&prev) => filed_int > prev,
                        None => true,
                    };
                    if !better { continue; }
                    best_quarter.insert(key, filed_int);

                    let qentry = quarterly_map.entry((cik, date))
                        .or_insert_with(|| QuarterlyIncome { date, ..Default::default() });
                    if filed_date > qentry.filed { qentry.filed = filed_date; }
                    match id {
                        15 => qentry.revenue = value,
                        16 => qentry.net_income = value,
                        17 => qentry.operating_cash_flow = value,
                        18 => qentry.operating_income = value,
                        19 => qentry.gross_profit = value,
                        20 => qentry.cost_of_revenue = value,
                        21 => qentry.interest_expense = value,
                        22 => qentry.income_tax = value,
                        23 => qentry.depreciation_amortization = value,
                        24 => qentry.rd_expense = value,
                        25 => qentry.sga_expense = value,
                        26 => qentry.eps_basic = value,
                        27 => qentry.eps_diluted = value,
                        _ => {}
                    }
                } else if is_annual {
                    let key = (cik, date, id);
                    let better = match best_annual.get(&key) {
                        Some(&prev) => filed_int > prev,
                        None => true,
                    };
                    if !better { continue; }
                    best_annual.insert(key, filed_int);

                    let aentry = annual_map.entry((cik, date))
                        .or_insert_with(|| FundamentalRecord { date, ..Default::default() });
                    if filed_date > aentry.filed { aentry.filed = filed_date; }
                    match id {
                        15 => aentry.revenue = value,
                        16 => aentry.net_income = value,
                        17 => aentry.operating_cash_flow = value,
                        18 => aentry.operating_income = value,
                        19 => aentry.gross_profit = value,
                        20 => aentry.cost_of_revenue = value,
                        21 => aentry.interest_expense = value,
                        22 => aentry.income_tax = value,
                        23 => aentry.depreciation_amortization = value,
                        24 => aentry.rd_expense = value,
                        25 => aentry.sga_expense = value,
                        26 => aentry.eps_basic = value,
                        27 => aentry.eps_diluted = value,
                        _ => {}
                    }
                }
            }
        }
    }

    // === Построение TTM из квартальных доходов ===
    let mut ttm_map: HashMap<(i64, NaiveDate), FundamentalRecord> = HashMap::new();
    let mut by_cik: HashMap<i64, Vec<QuarterlyIncome>> = HashMap::new();
    for ((cik, _), qi) in quarterly_map {
        by_cik.entry(cik).or_default().push(qi);
    }

    for (cik, mut quarters) in by_cik {
        quarters.sort_by_key(|q| q.date);
        // Дедупликация по дате — оставляем свежайший filed
        quarters.dedup_by(|a, b| a.date == b.date && a.filed <= b.filed);

        for i in 3..quarters.len() {
            let window = &quarters[i - 3..=i];
            let latest = &window[3];

            // Проверяем, что кварталы идут с шагом ~90 дней (60..120)
            let mut ok = true;
            for w in window.windows(2) {
                let gap = (w[1].date - w[0].date).num_days();
                if !(60..=120).contains(&gap) {
                    ok = false;
                    break;
                }
            }
            if !ok { continue; }

            let sum_f = |get: fn(&QuarterlyIncome) -> f64| -> f64 {
                let mut s = 0.0;
                let mut cnt = 0;
                for q in window {
                    let v = get(q);
                    if v.is_finite() {
                        s += v;
                        cnt += 1;
                    }
                }
                if cnt == 0 { f64::NAN } else { s }
            };

            let rec = FundamentalRecord {
                date: latest.date,
                filed: latest.filed,
                revenue: sum_f(|q| q.revenue),
                net_income: sum_f(|q| q.net_income),
                operating_cash_flow: sum_f(|q| q.operating_cash_flow),
                operating_income: sum_f(|q| q.operating_income),
                gross_profit: sum_f(|q| q.gross_profit),
                cost_of_revenue: sum_f(|q| q.cost_of_revenue),
                interest_expense: sum_f(|q| q.interest_expense),
                income_tax: sum_f(|q| q.income_tax),
                depreciation_amortization: sum_f(|q| q.depreciation_amortization),
                rd_expense: sum_f(|q| q.rd_expense),
                sga_expense: sum_f(|q| q.sga_expense),
                eps_basic: latest.eps_basic,
                eps_diluted: latest.eps_diluted,
                ..Default::default()
            };
            ttm_map.insert((cik, rec.date), rec);
        }
    }

    // === Доходы: сначала годовые, потом TTM поверх ===
    let mut income_map: HashMap<(i64, NaiveDate), FundamentalRecord> = HashMap::new();
    for ((cik, date), annual_rec) in annual_map {
        let mut rec = annual_rec.clone();
        rec.assets = f64::NAN;
        rec.liabilities = f64::NAN;
        rec.equity = f64::NAN;
        rec.cash = f64::NAN;
        rec.assets_current = f64::NAN;
        rec.liabilities_current = f64::NAN;
        rec.inventory = f64::NAN;
        rec.receivables = f64::NAN;
        rec.payables = f64::NAN;
        rec.shares_outstanding = f64::NAN;
        rec.long_term_debt = f64::NAN;
        rec.short_term_debt = f64::NAN;
        rec.total_debt = f64::NAN;
        rec.retained_earnings = f64::NAN;
        rec.shares_common = f64::NAN;
        income_map.insert((cik, date), rec);
    }
    for ((cik, date), ttm_rec) in ttm_map {
        income_map.insert((cik, date), ttm_rec);
    }

    // === Финальный merge: балансовые + доходные ===
    let mut all: HashMap<(i64, NaiveDate), FundamentalRecord> = HashMap::new();

    for ((cik, date), mut balance) in balance_map {
        if let Some(income) = income_map.remove(&(cik, date)) {
            balance.revenue = income.revenue;
            balance.net_income = income.net_income;
            balance.operating_cash_flow = income.operating_cash_flow;
            balance.operating_income = income.operating_income;
            balance.gross_profit = income.gross_profit;
            balance.cost_of_revenue = income.cost_of_revenue;
            balance.interest_expense = income.interest_expense;
            balance.income_tax = income.income_tax;
            balance.depreciation_amortization = income.depreciation_amortization;
            balance.rd_expense = income.rd_expense;
            balance.sga_expense = income.sga_expense;
            balance.eps_basic = income.eps_basic;
            balance.eps_diluted = income.eps_diluted;
            if income.filed > balance.filed { balance.filed = income.filed; }
        }
        all.insert((cik, date), balance);
    }

    // Оставшиеся доходные записи без баланса
    for (key, income) in income_map {
        all.insert(key, income);
    }

    // Раскладываем по тикерам
    let mut records: HashMap<String, Vec<FundamentalRecord>> = HashMap::new();
    for ((cik, _), rec) in all {
        if let Some(ticker) = tickers_map.get(&cik) {
            records.entry(ticker.clone()).or_default().push(rec);
        }
    }
    for vec in records.values_mut() {
        vec.sort_by_key(|r| (r.date, r.filed));
    }

    Ok(records)
}

// ====================== Производные показатели ======================

pub fn fundamental_features_on_date(
    _ticker: &str,
    records: &[FundamentalRecord],
    as_of: NaiveDate,
) -> Option<FundamentalFeatures> {
    // Доступные записи: filed <= as_of
    let available: Vec<&FundamentalRecord> = records.iter()
        .filter(|r| r.filed <= as_of)
        .collect();
    if available.is_empty() { return None; }

    // Последняя балансовая запись (по assets или equity)
    let last_balance = available.iter()
        .filter(|r| r.assets.is_finite() || r.equity.is_finite())
        .last()?;

    // Последняя запись с выручкой
    let last_revenue_rec = available.iter()
        .filter(|r| r.revenue.is_finite())
        .last();
    // Последняя запись с чистой прибылью
    let last_net_income_rec = available.iter()
        .filter(|r| r.net_income.is_finite())
        .last();

    if last_revenue_rec.is_none() && last_net_income_rec.is_none() {
        return None;
    }

    // Записи примерно за год назад (окно 300–430 дней)
    let prev_revenue = last_revenue_rec.and_then(|cur| {
        available.iter()
            .filter(|r| r.revenue.is_finite()
                && r.date <= cur.date - chrono::Duration::days(300)
                && r.date >= cur.date - chrono::Duration::days(430))
            .last()
    });
    let prev_net_income = last_net_income_rec.and_then(|cur| {
        available.iter()
            .filter(|r| r.net_income.is_finite()
                && r.date <= cur.date - chrono::Duration::days(300)
                && r.date >= cur.date - chrono::Duration::days(430))
            .last()
    });
    let prev_eps = last_net_income_rec.and_then(|cur| {
        available.iter()
            .filter(|r| r.eps_diluted.is_finite()
                && r.date <= cur.date - chrono::Duration::days(300)
                && r.date >= cur.date - chrono::Duration::days(430))
            .last()
    });

    let safe_div = |num: f64, den: f64| -> f64 {
        if den != 0.0 && num.is_finite() && den.is_finite() { num / den } else { f64::NAN }
    };

    // Извлекаем отдельные значения
    let revenue = last_revenue_rec.map(|r| r.revenue).unwrap_or(f64::NAN);
    let net_income = last_net_income_rec.map(|r| r.net_income).unwrap_or(f64::NAN);
    let gross_profit = last_revenue_rec.map(|r| r.gross_profit).unwrap_or(f64::NAN);
    let operating_income = last_net_income_rec
        .filter(|r| r.operating_income.is_finite())
        .map(|r| r.operating_income)
        .or_else(|| last_revenue_rec.filter(|r| r.operating_income.is_finite()).map(|r| r.operating_income))
        .unwrap_or(f64::NAN);
    let interest_expense = last_net_income_rec.map(|r| r.interest_expense).unwrap_or(f64::NAN);
    let rd_expense = last_revenue_rec.map(|r| r.rd_expense).unwrap_or(f64::NAN);
    let sga_expense = last_revenue_rec.map(|r| r.sga_expense).unwrap_or(f64::NAN);
    let eps_diluted = last_net_income_rec.map(|r| r.eps_diluted).unwrap_or(f64::NAN);

    // Производные показатели
    let roa = safe_div(net_income, last_balance.assets);
    let debt_to_equity = safe_div(last_balance.total_debt, last_balance.equity);
    let cash_ratio = safe_div(last_balance.cash, last_balance.liabilities_current);
    let inventory = if last_balance.inventory.is_nan() { 0.0 } else { last_balance.inventory };
    let quick_ratio = safe_div(last_balance.assets_current - inventory, last_balance.liabilities_current);
    let gross_margin = safe_div(gross_profit, revenue);
    let operating_margin = safe_div(operating_income, revenue);
    let net_margin = safe_div(net_income, revenue);
    let interest_coverage = safe_div(operating_income, interest_expense);
    let asset_turnover = safe_div(revenue, last_balance.assets);
    let rd_intensity = safe_div(rd_expense, revenue);
    let sga_intensity = safe_div(sga_expense, revenue);
    let roe = safe_div(net_income, last_balance.equity);
    let leverage = safe_div(last_balance.liabilities, last_balance.assets);
    let current_ratio = safe_div(last_balance.assets_current, last_balance.liabilities_current);
    let revenue_growth = prev_revenue
        .map(|p| safe_div(revenue - p.revenue, p.revenue))
        .unwrap_or(f64::NAN);
    let net_income_growth = prev_net_income
        .map(|p| safe_div(net_income - p.net_income, p.net_income))
        .unwrap_or(f64::NAN);
    let eps_growth = prev_eps
        .map(|p| safe_div(eps_diluted - p.eps_diluted, p.eps_diluted))
        .unwrap_or(f64::NAN);

    Some(FundamentalFeatures {
        roa, debt_to_equity, cash_ratio, quick_ratio,
        gross_margin, operating_margin, net_margin, interest_coverage,
        asset_turnover, rd_intensity, sga_intensity, eps_growth,
        roe, leverage, revenue_growth, net_income_growth, current_ratio,
    })
}

fn list_parquet_files(dir: &str) -> Result<Vec<PathBuf>> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map(|e| e.eq_ignore_ascii_case("parquet")).unwrap_or(false))
        .collect();
    v.sort();
    Ok(v)
}