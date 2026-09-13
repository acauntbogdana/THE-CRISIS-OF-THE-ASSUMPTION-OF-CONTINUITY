// Проверка торговых дней NYSE.

use std::collections::HashSet;
use chrono::NaiveDate;
use chrono::Datelike;

/// Пары (дата, описание) для дней, когда биржа полностью закрыта.
const NYSE_CLOSED_DAYS: &[(&str, &str)] = &[
    // 2018
    ("2018-01-01", "New Year's Day"),
    ("2018-01-15", "Martin Luther King's Birthday"),
    ("2018-02-19", "Washington's Birthday"),
    ("2018-03-30", "Good Friday"),
    ("2018-05-28", "Memorial Day"),
    ("2018-07-04", "Independence Day"),
    ("2018-09-03", "Labor Day"),
    ("2018-11-22", "Thanksgiving Day"),
    ("2018-12-05", "Exchange Holiday (national day of mourning)"),
    ("2018-12-25", "Christmas Day"),
    // 2019
    ("2019-01-01", "New Year's Day"),
    ("2019-01-21", "Martin Luther King's Birthday"),
    ("2019-02-18", "Washington's Birthday"),
    ("2019-04-19", "Good Friday"),
    ("2019-05-27", "Memorial Day"),
    ("2019-07-04", "Independence Day"),
    ("2019-09-02", "Labor Day"),
    ("2019-11-28", "Thanksgiving Day"),
    ("2019-12-25", "Christmas Day"),
    // 2020
    ("2020-01-01", "New Year's Day"),
    ("2020-01-20", "Martin Luther King's Birthday"),
    ("2020-02-17", "Washington's Birthday"),
    ("2020-04-10", "Good Friday"),
    ("2020-05-25", "Memorial Day"),
    ("2020-07-03", "Independence Day (Observed)"),
    ("2020-09-07", "Labor Day"),
    ("2020-11-26", "Thanksgiving Day"),
    ("2020-12-25", "Christmas Day"),
    // 2021
    ("2021-01-01", "New Year's Day"),
    ("2021-01-18", "Martin Luther King's Birthday"),
    ("2021-02-15", "Washington's Birthday"),
    ("2021-04-02", "Good Friday"),
    ("2021-05-31", "Memorial Day"),
    ("2021-07-05", "Independence Day (Observed)"),
    ("2021-09-06", "Labor Day"),
    ("2021-11-25", "Thanksgiving Day"),
    ("2021-12-24", "Christmas Day (Observed)"),
    // 2022
    ("2022-01-17", "Martin Luther King's Birthday"),
    ("2022-02-21", "Washington's Birthday"),
    ("2022-04-15", "Good Friday"),
    ("2022-05-30", "Memorial Day"),
    ("2022-06-20", "Juneteenth"),
    ("2022-07-04", "Independence Day"),
    ("2022-09-05", "Labor Day"),
    ("2022-11-24", "Thanksgiving Day"),
    ("2022-12-26", "Christmas Day (Observed)"),
];

/// Пары (дата, описание) для дней с укороченной торговой сессией (09:30–13:00).
const NYSE_SHORT_DAYS: &[(&str, &str)] = &[
    ("2018-07-03", "Independence Day (short day)"),
    ("2018-11-23", "Black Friday (short day)"),
    ("2018-12-24", "Christmas Eve (short day)"),
    ("2019-07-03", "Independence Day (short day)"),
    ("2019-11-29", "Black Friday (short day)"),
    ("2019-12-24", "Christmas Eve (short day)"),
    ("2020-11-27", "Black Friday (short day)"),
    ("2020-12-24", "Christmas Eve (short day)"),
    ("2021-11-26", "Black Friday (short day)"),
    ("2022-11-25", "Black Friday (short day)"),
];

/// Проверяет, является ли день торговым на NYSE (включая сокращённые дни).
pub fn is_nyse_trading_day(date: NaiveDate) -> bool {
    if matches!(date.weekday(), chrono::Weekday::Sat | chrono::Weekday::Sun) {
        return false;
    }
    let s = date.format("%Y-%m-%d").to_string();
    !NYSE_CLOSED_DAYS.iter().any(|(d, _)| *d == s)
}

/// Проверяет, является ли день сокращённым торговым днём (09:30–13:00).
pub fn is_nyse_short_day(date: NaiveDate) -> bool {
    let s = date.format("%Y-%m-%d").to_string();
    NYSE_SHORT_DAYS.iter().any(|(d, _)| *d == s)
}

/// Возвращает описание дня, если он является праздником или коротким днём.
pub fn day_description(date: NaiveDate) -> Option<&'static str> {
    let s = date.format("%Y-%m-%d").to_string();
    if let Some((_, desc)) = NYSE_CLOSED_DAYS.iter().find(|(d, _)| *d == s) {
        return Some(desc);
    }
    if let Some((_, desc)) = NYSE_SHORT_DAYS.iter().find(|(d, _)| *d == s) {
        return Some(desc);
    }
    None
}

/// Проверить наличие файлов за все торговые дни периода.
pub fn verify_nyse_calendar(processed_files: &HashSet<String>) {
    let start = NaiveDate::from_ymd_opt(2018, 1, 1).unwrap();
    let end = NaiveDate::from_ymd_opt(2022, 12, 31).unwrap();
    let mut missing = Vec::new();
    let mut extra = Vec::new();

    let mut d = start;
    while d <= end {
        let fname = format!("{}.parquet", d.format("%Y-%m-%d"));
        if is_nyse_trading_day(d) {
            if !processed_files.contains(&fname) {
                missing.push(d);
            }
        } else {
            if processed_files.contains(&fname) {
                extra.push(d);
            }
        }
        d += chrono::Duration::days(1);
    }

    if !missing.is_empty() {
        println!("ВНИМАНИЕ: отсутствуют файлы за {} торговых дней:", missing.len());
        for md in missing.iter().take(20) {
            let desc = day_description(*md).unwrap_or("(нет описания)");
            println!("  {} - {}", md.format("%Y-%m-%d"), desc);
        }
        if missing.len() > 20 {
            println!("  ... и ещё {}", missing.len() - 20);
        }
    } else {
        println!("Все торговые дни покрыты файлами.");
    }

    if !extra.is_empty() {
        println!("ВНИМАНИЕ: обнаружены файлы за {} неторговых дней:", extra.len());
        for ed in extra.iter().take(20) {
            let desc = day_description(*ed).unwrap_or("(нет описания)");
            println!("  {} - {}", ed.format("%Y-%m-%d"), desc);
        }
        if extra.len() > 20 {
            println!("  ... и ещё {}", extra.len() - 20);
        }
    } else {
        println!("Лишних файлов за неторговые дни не найдено.");
    }
}