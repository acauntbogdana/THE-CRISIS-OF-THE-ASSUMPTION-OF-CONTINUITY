//! Основной анализ: уровни I-V и логистическая регрессия.
use std::{
    collections::{BTreeMap, HashMap},
    fs::OpenOptions,
    io::Write,
    time::Instant,
};

use anyhow::{anyhow, Context, Result};
use chrono::Datelike;
use ndarray::{Array1, Array2};

// Для потоковой записи/чтения панели на диск (вместо накопления в ОЗУ)
use csv::ReaderBuilder;
use std::fs::File;
use std::io::BufReader;

use arrow::array::{Array, Int64Array, LargeStringArray};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use crate::{
    bars_io::{list_available_tickers, load_bars_for_ticker},
    config::{
        HORIZONS, JUMP_THRESHOLD_MULT, MIN_VOLUME, OUTPUT_DIR,
        SELECTED_MACRO_COLS, STSRV_PARAMS,
    },
    data_loading::{unix_day_to_date, BarAgg},
    equations::*,
    fundamental_data::{
        fundamental_features_on_date, load_all_fundamental_records, FUNDAMENTAL_FEATURE_COUNT,
    },
    macro_data::load_macro_ffill,
       statistics::{
        mean, median, bootstrap_median_ci, bootstrap_median_less_than,
    },
};

pub fn run() -> Result<()> {
    std::fs::create_dir_all(OUTPUT_DIR)?;
    let start = Instant::now();

    // Шаг 1: Построение баров
    println!("Шаг 1: Построение баров для всех тикеров...");
    crate::data_loading::build_all_bars()?;

    // Проверка календаря
    let state_file = format!("{}/state.json", OUTPUT_DIR);
    let state: ProcessingState = if Path::new(&state_file).exists() {
        let data = std::fs::read_to_string(&state_file)?;
        serde_json::from_str(&data)?
    } else {
        ProcessingState::default()
    };
    verify_nyse_calendar(&state.processed_files);

    // Шаг 2: Отбор тикеров по ликвидности
    println!("Шаг 2: Отбор тикеров по ликвидности...");
    let tickers = list_available_tickers()?;
    println!("Всего тикеров с барами: {}", tickers.len());

    let mut selected_tickers = Vec::new();
    for ticker in &tickers {
        let bars = load_bars_for_ticker(ticker)?;
        let total_volume: u64 = bars.iter().map(|b| b.volume).sum();
        if total_volume >= MIN_VOLUME {
            selected_tickers.push(ticker.clone());
        }
    }
    println!("Отобрано тикеров после фильтра MIN_VOLUME: {}", selected_tickers.len());

    // Шаг 3: Загрузка справочников и фундаментальных данных
    println!("Шаг 3: Загрузка справочников и фундаментальных данных...");
    let tickers_map = load_tickers_map()?;
    let fundamental_records = load_all_fundamental_records(&tickers_map)?;
    let macro_data = load_macro_ffill()?;

    // >>> ДИАГНОСТИКА ФУНДАМЕНТАЛЬНЫХ ДАННЫХ
    println!("DEBUG: fundamental_records: {} тикеров", fundamental_records.len());
       if let Some(recs) = fundamental_records.get("A") {
        println!(
            "DEBUG: A -> {} записей; первая date={} filed={}",
            recs.len(),
            recs.first().map(|r| r.date.to_string()).unwrap_or_default(),
            recs.first().map(|r| r.filed.to_string()).unwrap_or_default(),
        );
        for (i, r) in recs.iter().enumerate().take(30) {
            println!(
                "DEBUG: A[{}] date={} filed={} assets={:.0} equity={:.0} revenue={:.0} net_income={:.0} roa?={}",
                i, r.date, r.filed,
                if r.assets.is_finite() { r.assets } else { f64::NAN },
                if r.equity.is_finite() { r.equity } else { f64::NAN },
                if r.revenue.is_finite() { r.revenue } else { f64::NAN },
                if r.net_income.is_finite() { r.net_income } else { f64::NAN },
                if r.assets.is_finite() && r.net_income.is_finite() { "ок" } else { "NaN" },
            );
        }
    } else {
        println!("DEBUG: A отсутствует в fundamental_records");
    }
    // <<< ДИАГНОСТИКА

    // Шаг 4: Подготовка отчёта
    let mut report = OpenOptions::new()
        .create(true)
        .append(true)
        .open(format!("{}/report.txt", OUTPUT_DIR))?;
    writeln!(report, "\n=== Полный анализ (все тикеры) от {} ===", chrono::Utc::now())?;

    // Заголовки признаков и путь к панели — нужны для чтения панели
    let mut panel_feature_names: Vec<String> = vec![
        "const".to_string(),
        "spread".to_string(),
        "ln(vol)".to_string(),
        "sigma_intraday".to_string(),
    ];
    panel_feature_names.extend(SELECTED_MACRO_COLS.iter().map(|s| s.to_string()));
    panel_feature_names.extend(
        [
            "roa", "debt_to_equity", "cash_ratio", "quick_ratio", "gross_margin",
            "operating_margin", "net_margin", "interest_coverage", "asset_turnover",
            "rd_intensity", "sga_intensity", "eps_growth",
            "roe", "leverage", "rev_growth", "ni_growth", "current_ratio",
        ].iter().map(|s| s.to_string()),
    );

    // Шаг 5: Уровни I, II, III, V (последовательно по тикерам, агрегируем статистики)
    println!("Шаг 5: Вычисление уровней I, II, III, V...");
    let mut beta_values = Vec::new();
    let mut alpha_by_h: Vec<Vec<f64>> = vec![Vec::new(); HORIZONS.len()];
    let mut cusum_mu_pvals = Vec::new();
    let mut cusum_sigma_pvals = Vec::new();
    let mut delta_alpha_vals = Vec::new();
    let mut stsrv_estimates = Vec::new();
    let mut economic_significant_jumps = 0usize;
    let mut total_jumps = 0usize;

    // Панель не накапливается в памяти, пишется на диск.
    // Имена признаков фиксированы заранее и записываются в заголовок.
    let mut panel_feature_names: Vec<String> = vec![
        "const".to_string(),
        "spread".to_string(),
        "ln(vol)".to_string(),
        "sigma_intraday".to_string(),
    ];
    panel_feature_names.extend(SELECTED_MACRO_COLS.iter().map(|s| s.to_string()));
    panel_feature_names.extend(
        [
            "roa", "debt_to_equity", "cash_ratio", "quick_ratio", "gross_margin",
            "operating_margin", "net_margin", "interest_coverage", "asset_turnover",
            "rd_intensity", "sga_intensity", "eps_growth",
            "roe", "leverage", "rev_growth", "ni_growth", "current_ratio",
        ]
        .iter()
        .map(|s| s.to_string()),
    );
    let n_features = panel_feature_names.len();

    let panel_path = format!("{}/panel_data.csv", OUTPUT_DIR);
    /*
    let panel_file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&panel_path)
        .with_context(|| format!("Не удалось создать файл панели {}", panel_path))?;
    let mut panel_writer = WriterBuilder::new().from_writer(BufWriter::new(panel_file));
    {
        // ticker и date пишем как обычные колонки — по ним восстановим paper_id/date_id при чтении
        let mut header: Vec<String> = vec!["ticker".to_string(), "date".to_string()];
        header.extend(panel_feature_names.iter().cloned());
        header.push("y".to_string());
        panel_writer.write_record(&header)?;
    }
    */
    let mut panel_rows_written: u64 = 0;
    let mut panel_rows_skipped: u64 = 0;

    for (ticker_idx, ticker) in selected_tickers.iter().enumerate() {
        // Бары для тикера загружаются заново и живут только в рамках этой итерации
        let bars = load_bars_for_ticker(ticker)?;
        if bars.len() < 100 { continue; }
        let closes: Vec<f64> = bars.iter().map(|b| b.close).collect();

        // Уровень I: бета и альфа по горизонтам
        let mut kurtoses = Vec::new();
        let mut horizons = Vec::new();
        for (i, &h) in HORIZONS.iter().enumerate() {
            let returns: Vec<f64> = (0..closes.len().saturating_sub(h))
                .filter_map(|t| agg_log_return_from_prices(&closes, t, h))
                .collect();
            if returns.len() < 100 { continue; }
            let kurt = empirical_kurtosis(&returns).unwrap_or(f64::NAN);
            let mut sorted = returns.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let alpha = tail_index(&sorted).unwrap_or(f64::NAN);
            if alpha.is_finite() && kurt.is_finite() && kurt > 0.0 {
                alpha_by_h[i].push(alpha);
            }
            if kurt > 0.0 {
                kurtoses.push(kurt);
                horizons.push(h as f64);
            }
        }
        if horizons.len() >= 2 {
            if let Some((beta, _)) = estimate_beta(&horizons, &kurtoses) {
                beta_values.push(beta);
            }
        }

        // Уровень II: CUSUM и MF-DFA
        let returns: Vec<f64> = closes.windows(2).map(|w| (w[1] / w[0]).ln()).collect();
        if returns.len() >= 200 {
            let (_, p_mu) = cusum_bwb_mean_test(&returns, None, 200);
            let (_, p_sigma) = cusum_bwb_variance_test(&returns, None, 200);
            cusum_mu_pvals.push(p_mu);
            cusum_sigma_pvals.push(p_sigma);
            let mf = mf_dfa(&returns, -5.0, 5.0, 0.5, None, 1);
            delta_alpha_vals.push(mf.delta_alpha);
        }

        // Уровень III: S-TSRV
        let log_prices: Vec<f64> = closes.iter().map(|&p| p.ln()).collect();
        let day_len = 390; // количество минут в торговом дне (предполагается минутные бары)
        let mut day_start = 0;
        while day_start + day_len <= log_prices.len() {
            let day_slice = &log_prices[day_start..day_start + day_len];
            let smoothed = preaverage_and_truncate(day_slice, &STSRV_PARAMS);
            if smoothed.len() > STSRV_PARAMS.k {
                let est = truncated_stsrv(&smoothed, &smoothed, &STSRV_PARAMS);
                stsrv_estimates.push(est);
            }
            day_start += day_len;
        }

        // Уровень V и подготовка панели (Уровень IV)
        let mut day_groups: BTreeMap<i64, Vec<&BarAgg>> = BTreeMap::new();
        for bar in &bars {
            day_groups.entry(bar.date).or_default().push(bar);
        }
        for (day_unix, day_bars) in day_groups {
            if day_bars.len() < 100 { continue; }
            let date = unix_day_to_date(day_unix);
            if date.year() < 2019 { continue; }

            let mut five_min_returns: Vec<(f64, f64, f64)> = Vec::new();
            // Используем только полные 5-минутные интервалы
            for chunk in day_bars.chunks(5) {
                if chunk.len() != 5 { continue; }
                let first = chunk[0];
                let last = chunk[4];
                let ret = (last.close / first.close).ln();
                // Если нет bid/ask — ставим 0.0, чтобы строка не отбрасывалась
                let spread = match (last.best_ask, last.best_bid) {
                    (Some(ask), Some(bid)) => (ask - bid) / last.close,
                    _ => 0.0,
                };
                let vol = chunk.iter().map(|b| b.volume).sum::<u64>() as f64;
                five_min_returns.push((ret, spread, vol));
            }
            if five_min_returns.is_empty() { continue; }
            let rets_only: Vec<f64> = five_min_returns.iter().map(|r| r.0).collect();
            let sigma = variance(&rets_only).sqrt();
            let mut has_jump = false;
            for (ret, spread, _vol) in &five_min_returns {
                if ret.abs() > JUMP_THRESHOLD_MULT * sigma {
                    has_jump = true;
                    total_jumps += 1;
                    // Экономическая значимость: phi = |ret| / spread
                                        if let Some(phi) = economic_filter_phi(ret.abs(), *spread) {
                        if phi > 1.0 {
                            economic_significant_jumps += 1;
                        }
                    }
                }
            }
            let y_val: f64 = if has_jump { 1.0 } else { 0.0 };

            let day_close = day_bars.last().unwrap().close;

            // Ищем последний бар дня с валидными bid/ask
            let mut spread_eod = f64::NAN;
            for bar in day_bars.iter().rev() {
                if let (Some(ask), Some(bid)) = (bar.best_ask, bar.best_bid) {
                    spread_eod = (ask - bid) / day_close;
                    break;
                }
            }
            // Если не нашли — берём среднее по 5-минутным интервалам
            if !spread_eod.is_finite() {
                let mut sum = 0.0;
                let mut cnt = 0usize;
                for chunk in day_bars.chunks(5) {
                    if chunk.len() != 5 { continue; }
                    let last = chunk[4];
                    if let (Some(ask), Some(bid)) = (last.best_ask, last.best_bid) {
                        sum += (ask - bid) / last.close;
                        cnt += 1;
                    }
                }
                spread_eod = if cnt > 0 { sum / cnt as f64 } else { 0.0 };
            }

            let volume = day_bars.iter().map(|b| b.volume).sum::<u64>() as f64;
            let ln_volume = if volume > 0.0 { volume.ln() } else { 0.0 };

            // const, spread, ln(vol), sigma_intraday
            let mut row = vec![1.0, spread_eod, ln_volume, sigma];

            // Макро-переменные
            match macro_data.get(&date) {
                Some(macro_row) => {
                    for key in SELECTED_MACRO_COLS {
                        let value = macro_row
                            .get(*key)
                            .copied()
                            .flatten()
                            .unwrap_or(f64::NAN);
                        row.push(value);
                    }
                }
                None => {
                    row.extend(std::iter::repeat(f64::NAN).take(SELECTED_MACRO_COLS.len()));
                }
            }

            // Фундаментальные показатели
            let fund = fundamental_records
                .get(ticker)
                .and_then(|recs| fundamental_features_on_date(ticker, recs, date));
            match fund {
                Some(f) => {
                    row.push(f.roa);
                    row.push(f.debt_to_equity);
                    row.push(f.cash_ratio);
                    row.push(f.quick_ratio);
                    row.push(f.gross_margin);
                    row.push(f.operating_margin);
                    row.push(f.net_margin);
                    row.push(f.interest_coverage);
                    row.push(f.asset_turnover);
                    row.push(f.rd_intensity);
                    row.push(f.sga_intensity);
                    row.push(f.eps_growth);
                    row.push(f.roe);
                    row.push(f.leverage);
                    row.push(f.revenue_growth);
                    row.push(f.net_income_growth);
                    row.push(f.current_ratio);
                }
                None => {
                    row.extend(std::iter::repeat(f64::NAN).take(FUNDAMENTAL_FEATURE_COUNT));
                }
            }

            // Проверка на конечность непосредственно перед записью
            if row.len() == n_features && row.iter().all(|v| v.is_finite()) && y_val.is_finite() {
                let mut record: Vec<String> = Vec::with_capacity(2 + n_features + 1);
                record.push(ticker.clone());
                record.push(date.format("%Y-%m-%d").to_string());
                for v in &row {
                    record.push(v.to_string());
                }
                record.push(y_val.to_string());
                //panel_writer.write_record(&record)?; закомментировано
                panel_rows_written += 1;
            } else {
                if panel_rows_skipped < 3 {
                    let na_names: Vec<String> = row.iter().enumerate()
                        .filter(|(_, v)| !v.is_finite())
                        .map(|(i, _)| panel_feature_names.get(i).cloned().unwrap_or_else(|| format!("col{}", i)))
                        .take(10)
                        .collect();
                    eprintln!(
                        "DEBUG: пропущена строка (date={}, ticker={}): y={}, non-finite={}, первые: {:?}",
                        date, ticker, y_val,
                        row.iter().filter(|v| !v.is_finite()).count(),
                        na_names
                    );
                }
                panel_rows_skipped += 1;
            }
        }

        if (ticker_idx + 1) % 100 == 0 {
            //panel_writer.flush()?; закомментировано
            println!(
                "Обработано тикеров: {}/{}, строк панели записано: {}",
                ticker_idx + 1,
                selected_tickers.len(),
                panel_rows_written
            );
        }
    }
   // panel_writer.flush()?; закомментировано
  //  drop(panel_writer);
    println!(
        "Панель сформирована на диске: {} строк -> {}",
        panel_rows_written, panel_path
    );

    // Логирование доли отброшенных строк
    if panel_rows_written + panel_rows_skipped > 0 {
        let skip_ratio = panel_rows_skipped as f64 / (panel_rows_written + panel_rows_skipped) as f64;
        writeln!(
            report,
            "Панель: всего строк {}, записано {}, пропущено (NaN/Inf) {} ({:.2}%)",
            panel_rows_written + panel_rows_skipped,
            panel_rows_written,
            panel_rows_skipped,
            skip_ratio * 100.0
        )?;
    }

        // Запись агрегированных статистик
    writeln!(report, "\n--- Уровень I ---")?;
    let med_beta = median(&beta_values);
    let n_beta = beta_values.len();
    let (se_beta, ci_lo, ci_hi) =
        bootstrap_median_ci(&beta_values, 2000, 0.05);
    let p_less_one =
        bootstrap_median_less_than(&beta_values, 1.0, 2000);
    let frac_below_one = if n_beta > 0 {
        beta_values.iter().filter(|&&b| b < 1.0).count() as f64
            / n_beta as f64
    } else {
        f64::NAN
    };
    writeln!(
        report,
        "Медианная бета по рынку: {:.3} (n тикеров = {})",
        med_beta, n_beta
    )?;
    writeln!(
        report,
        "SE(медианы) = {:.4}; 95% ДИ = [{:.3}, {:.3}]",
        se_beta, ci_lo, ci_hi
    )?;
    writeln!(
        report,
        "Односторонний bootstrap-тест H0: median >= 1 vs H1: median < 1: p = {:.4}",
        p_less_one
    )?;
    writeln!(
        report,
        "Доля тикеров с beta < 1: {:.2}%",
        frac_below_one * 100.0
    )?;
    for (i, &h) in HORIZONS.iter().enumerate() {
        let med_alpha = median(&alpha_by_h[i]);
        writeln!(report, "h = {}: медианный alpha = {:.3}", h, med_alpha)?;
    }

    writeln!(report, "\n--- Уровень II ---")?;
    writeln!(report, "Медианный p-value CUSUM mean: {:.3}", median(&cusum_mu_pvals))?;
    writeln!(report, "Медианный p-value CUSUM sigma: {:.3}", median(&cusum_sigma_pvals))?;
    writeln!(report, "Медианный Δα MF-DFA: {:.3}", median(&delta_alpha_vals))?;

    writeln!(report, "\n--- Уровень III ---")?;
    writeln!(report, "Средняя S-TSRV по всем тикерам и дням: {:.6}", mean(&stsrv_estimates))?;
    let xi = STSRV_PARAMS.xi();
    let n_est = stsrv_estimates.len() as f64;
    let se = (2.0 * xi / 3.0).sqrt() / n_est.sqrt();
    writeln!(report, "Приблизительный 95% ДИ: [{:.6}, {:.6}]",
        mean(&stsrv_estimates) - 1.96*se, mean(&stsrv_estimates) + 1.96*se)?;

    writeln!(report, "\n--- Уровень V ---")?;
    writeln!(report, "Всего внутридневных скачков: {}", total_jumps)?;
    writeln!(report, "Экономически значимых (Φ>1): {}", economic_significant_jumps)?;




            // Шаг 6: Логистическая регрессия (Уровень IV)
    println!("Шаг 6: Чтение панели с диска ({})...", panel_path);
    let (x_arr, y_arr, paper_ids, date_ids, names) =
        read_panel_csv(&panel_path, &panel_feature_names)?;
    println!("Загружено {} наблюдений для логистической регрессии", x_arr.nrows());

    if x_arr.nrows() > 50 {
        // --- 1. Исключаем sigma_intraday (индекс 3) ---
        const DROP_SIGMA_IDX: usize = 3; // const=0, spread=1, ln(vol)=2, sigma=3
        let n_keep = x_arr.ncols() - 1;
        let mut x_clean = Array2::<f64>::zeros((x_arr.nrows(), n_keep));
        let mut names_clean: Vec<String> = Vec::with_capacity(n_keep);
        let mut col_out = 0;
        for j in 0..x_arr.ncols() {
            if j == DROP_SIGMA_IDX { continue; }
            for i in 0..x_arr.nrows() {
                x_clean[[i, col_out]] = x_arr[[i, j]];
            }
            names_clean.push(names[j].clone());
            col_out += 1;
        }
        println!("DEBUG: sigma_intraday исключён из регрессии (функциональная связь с целевой)");

        // --- 2. Стандартизация (кроме const, индекс 0) ---
        let mut x_std = x_clean.clone();
        for j in 1..x_std.ncols() {
            let col = x_std.column(j);
            let mean_val = col.mean().unwrap_or(0.0);
            let std_val = col.std(0.0);
            if std_val > 1e-12 {
                for i in 0..x_std.nrows() {
                    x_std[[i, j]] = (x_std[[i, j]] - mean_val) / std_val;
                }
            }
        }
        println!("DEBUG: признаки стандартизированы (кроме const)");

        // --- 3. Регрессия ---
        let p = x_std.ncols();
        let (beta, hessian, scores) = fit_logistic(x_std.view(), y_arr.view());
        let cov = cluster_cov_twoway(&paper_ids, &date_ids, scores.view(), hessian.view());

        // Метрики качества модели
        let p_hat = predict_proba(x_std.view(), beta.view());
        let auc_val = auc_score(y_arr.view(), p_hat.view());
        let r2_mcf = pseudo_r2_mcfadden(x_std.view(), y_arr.view(), beta.view());
                // === AUC для вложенных моделей ===
        // Границы блоков в x_std (после удаления sigma):
        //   0                — const
        //   1..=2            — микроструктура (spread, ln(vol))
        //   3..=3+n_macro-1  — макро
        //   далее            — фундаментальные
        let n_macro = SELECTED_MACRO_COLS.len();
        let n_fund = FUNDAMENTAL_FEATURE_COUNT;
        let total_cols = x_std.ncols();

        let micro_end = 3usize.min(total_cols);
        let macro_end = (3 + n_macro).min(total_cols);
        let fund_end = (3 + n_macro + n_fund).min(total_cols);

        let subsets: Vec<(&str, usize)> = vec![
            ("const_only",  1),
            ("micro",       micro_end),
            ("micro_macro", macro_end),
            ("full",        fund_end),
        ];

        let auc_path = format!("{}/auc_summary.csv", OUTPUT_DIR);
        let mut auc_file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&auc_path)?;
        writeln!(auc_file, "subset,n_features,auc,r2_mcfadden")?;

        for (name, n_feat) in &subsets {
            if *n_feat > total_cols || *n_feat == 0 {
                continue;
            }
            let x_sub = x_std.slice(ndarray::s![.., ..*n_feat]).to_owned();
            let (beta_sub, _, _) = fit_logistic(x_sub.view(), y_arr.view());
            let p_sub = predict_proba(x_sub.view(), beta_sub.view());
            let auc_sub = auc_score(y_arr.view(), p_sub.view());
            let r2_sub = pseudo_r2_mcfadden(x_sub.view(), y_arr.view(), beta_sub.view());
            writeln!(
                auc_file,
                "{},{},{:.4},{:.4}",
                name, n_feat, auc_sub, r2_sub
            )?;
            writeln!(
                report,
                "AUC[{}, p={}] = {:.4}; pseudo-R² = {:.4}",
                name, n_feat, auc_sub, r2_sub
            )?;
        }
        drop(auc_file);

        writeln!(report, "\n--- Уровень IV: Логистическая регрессия (кластеризация по бумаге и дате) ---")?;
        writeln!(report, "Примечание: признаки стандартизированы (кроме const); sigma_intraday исключён.")?;
        writeln!(report, "Коэффициенты = изменение лог-шансов при росте признака на 1σ.")?;
        writeln!(report, "AUC = {:.4}", auc_val)?;
        writeln!(report, "Pseudo-R² (McFadden) = {:.4}", r2_mcf)?;

        if names_clean.len() != p {
            return Err(anyhow!(
                "Число имён ({}) не совпадает с числом колонок X ({}).",
                names_clean.len(), p
            ));
        }

        for j in 0..p {
            let se = cov[[j, j]].sqrt();
            let name = names_clean.get(j).map(|s| s.as_str()).unwrap_or("?");
            writeln!(
                report,
                "{:<15}: coef={:8.4}, se={:8.4}, z={:7.3}",
                name, beta[j], se, beta[j] / se
            )?;
        }
    } else {
        writeln!(report, "Недостаточно наблюдений в панели (<=50)")?;
    }

    // Шаг 7: Графики
    println!("Шаг 7: Генерация графиков...");
    crate::plots::generate_plots(&selected_tickers)?;

    println!("Готово за {:.1} с", start.elapsed().as_secs_f64());
    Ok(())
}

/// Читает панель из CSV, созданного в run().
/// Возвращает (матрица X, вектор y, paper_ids, date_ids, имена признаков).
fn read_panel_csv(
    path: &str,
    feature_names: &[String],
) -> Result<(Array2<f64>, Array1<f64>, Vec<usize>, Vec<usize>, Vec<String>)> {
    let file = File::open(path)
        .with_context(|| format!("Не удалось открыть файл панели {}", path))?;
    let reader = BufReader::new(file);
    let mut rdr = ReaderBuilder::new().has_headers(true).from_reader(reader);

    let mut x_rows: Vec<Vec<f64>> = Vec::new();
    let mut y: Vec<f64> = Vec::new();
    let mut paper_ids: Vec<usize> = Vec::new();
    let mut date_ids: Vec<usize> = Vec::new();
    let mut paper_map: HashMap<String, usize> = HashMap::new();
    let mut date_map: HashMap<String, usize> = HashMap::new();
    let mut next_paper_id = 0usize;
    let mut next_date_id = 0usize;

    for record in rdr.records() {
        let rec = record?;
        // Структура CSV: ticker, date, [feature_names...], y
        let ticker = rec.get(0).unwrap_or("").to_string();
        let date_str = rec.get(1).unwrap_or("").to_string();
        let n_features = feature_names.len();
        if rec.len() < 2 + n_features + 1 {
            continue; // повреждённая строка
        }

        let mut features = Vec::with_capacity(n_features);
        for j in 0..n_features {
            let val: f64 = rec
                .get(2 + j)
                .and_then(|s| s.parse().ok())
                .unwrap_or(f64::NAN);
            features.push(val);
        }
        let target: f64 = rec
            .get(2 + n_features)
            .and_then(|s| s.parse().ok())
            .unwrap_or(f64::NAN);

        if features.iter().any(|v| !v.is_finite()) || !target.is_finite() {
            continue;
        }

        let paper_id = *paper_map.entry(ticker.clone()).or_insert_with(|| {
            let id = next_paper_id;
            next_paper_id += 1;
            id
        });
        let date_id = *date_map.entry(date_str.clone()).or_insert_with(|| {
            let id = next_date_id;
            next_date_id += 1;
            id
        });

        x_rows.push(features);
        y.push(target);
        paper_ids.push(paper_id);
        date_ids.push(date_id);
    }

    if x_rows.is_empty() {
        return Err(anyhow!("Панель пуста после чтения CSV"));
    }

    let n = x_rows.len();
    let p = x_rows[0].len();
    let mut x_arr = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        for j in 0..p {
            x_arr[[i, j]] = x_rows[i][j];
        }
    }
    let y_arr = Array1::from(y);

    Ok((x_arr, y_arr, paper_ids, date_ids, feature_names.to_vec()))
}

// Загрузка справочника тикеров (CIK -> ticker)
fn load_tickers_map() -> Result<BTreeMap<i64, String>> {
    let file = File::open(crate::config::TICKERS_FILE)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let schema = builder.schema().clone();
    let reader = builder.with_batch_size(1024).build()?;

    // Исправлено: schema.index_of возвращает Result, используем map_err
    let idx_cik = schema.index_of("cik").map_err(|_| anyhow!("нет cik"))?;
    let idx_ticker = schema.index_of("ticker").map_err(|_| anyhow!("нет ticker"))?;

    let mut map = BTreeMap::new();
    for batch in reader {
        let batch = batch?;
        let cik_arr = batch.column(idx_cik).as_any().downcast_ref::<Int64Array>()
            .ok_or_else(|| anyhow!("cik не Int64Array"))?;
        // Исправлено: LargeUtf8Array -> LargeStringArray
        let ticker_arr = batch.column(idx_ticker).as_any().downcast_ref::<LargeStringArray>()
            .ok_or_else(|| anyhow!("ticker не LargeStringArray"))?;

        for i in 0..batch.num_rows() {
            if cik_arr.is_valid(i) && ticker_arr.is_valid(i) {
                let cik = cik_arr.value(i);
                let ticker = ticker_arr.value(i).to_string();
                map.insert(cik, ticker);
            }
        }
    }
    Ok(map)
}