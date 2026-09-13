use anyhow::{bail, Result};
use plotters::prelude::*;
use plotters::style::FontStyle;
use plotters::coord::Shift;

use crate::{
    bars_io::load_bars_for_ticker,
    config::{HORIZONS, OUTPUT_DIR},
    equations::{
        agg_log_return_from_prices,
        empirical_kurtosis,
        mf_dfa,
        tail_index,
    },
    statistics::median,
};

const IMG_WIDTH: u32 = 1400;
const IMG_HEIGHT: u32 = 1100;
const MAIN_LINE_WIDTH: u32 = 4;
const IQR_LINE_WIDTH: u32 = 2;
const FONT_SIZE: f64 = 46.0;
const MARKER_SIZE: i32 = 5;

const MIN_RETURN_OBSERVATIONS: usize = 100;
const MIN_MFDFA_OBSERVATIONS: usize = 1000;
const MIN_CROSS_SECTIONAL_N: usize = 10;
const MIN_SPECTRUM_STOCKS: usize = 10;
const MFDFA_GRID_POINTS: usize = 201;
const MIN_SPECTRUM_POINTS: usize = 3;

// Ограничения для корректного отображения MF-DFA спектра
const MFDFA_ALPHA_MIN: f64 = 0.0;      // нижняя граница α
const MFDFA_ALPHA_MAX: f64 = 3.0;      // верхняя граница α
const MFDFA_MIN_TICKERS_PER_POINT: usize = 50;  // минимум тикеров в точке медианы

// ---------------------------------------------------------------------------
// Вспомогательные структуры и функции
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
struct SummaryStats {
    median: f64,
    q25: f64,
    q75: f64,
}

#[derive(Clone, Copy, Debug)]
struct SpectrumPoint {
    alpha: f64,
    median: f64,
    q25: f64,
    q75: f64,
}

fn quantile_sorted(sorted: &[f64], probability: f64) -> Option<f64> {
    if sorted.is_empty() {
        return None;
    }
    if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
        return None;
    }
    let n = sorted.len();
    if n == 1 {
        return Some(sorted[0]);
    }
    let position = probability * (n - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    if lower == upper {
        return Some(sorted[lower]);
    }
    let weight = position - lower as f64;
    Some(sorted[lower] + weight * (sorted[upper] - sorted[lower]))
}

fn summary_stats(values: &[f64]) -> Option<SummaryStats> {
    let mut sorted: Vec<f64> = values
        .iter()
        .copied()
        .filter(|x| x.is_finite())
        .collect();
    if sorted.len() < MIN_CROSS_SECTIONAL_N {
        return None;
    }
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median_value = median(&sorted);
    let q25 = quantile_sorted(&sorted, 0.25)?;
    let q75 = quantile_sorted(&sorted, 0.75)?;
    if !median_value.is_finite() || !q25.is_finite() || !q75.is_finite() {
        return None;
    }
    Some(SummaryStats {
        median: median_value,
        q25,
        q75,
    })
}

fn positive_range(values: &[f64]) -> Option<(f64, f64)> {
    let valid: Vec<f64> = values
        .iter()
        .copied()
        .filter(|x| x.is_finite() && *x > 0.0)
        .collect();
    if valid.is_empty() {
        return None;
    }
    let min_value = valid.iter().copied().fold(f64::INFINITY, f64::min);
    let max_value = valid.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !min_value.is_finite() || !max_value.is_finite() {
        return None;
    }
    if (max_value - min_value).abs() < f64::EPSILON {
        let lower = (min_value * 0.8).max(1e-12);
        let upper = (max_value * 1.2).max(lower * 1.01);
        return Some((lower, upper));
    }
    Some(((min_value * 0.85).max(1e-12), max_value * 1.15))
}

fn linear_range(values: &[f64]) -> Option<(f64, f64)> {
    let valid: Vec<f64> = values
        .iter()
        .copied()
        .filter(|x| x.is_finite())
        .collect();
    if valid.is_empty() {
        return None;
    }
    let min_value = valid.iter().copied().fold(f64::INFINITY, f64::min);
    let max_value = valid.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !min_value.is_finite() || !max_value.is_finite() {
        return None;
    }
    if (max_value - min_value).abs() < f64::EPSILON {
        let padding = if min_value.abs() > 1e-12 { min_value.abs() * 0.15 } else { 1.0 };
        return Some((min_value - padding, max_value + padding));
    }
    let padding = (max_value - min_value) * 0.10;
    Some((min_value - padding, max_value + padding))
}

fn prepare_mfdfa_spectrum(alpha_q: &[f64], f_alpha: &[f64]) -> Vec<(f64, f64)> {
    let mut spectrum: Vec<(f64, f64)> = alpha_q
        .iter()
        .copied()
        .zip(f_alpha.iter().copied())
        .filter(|(alpha, f)| alpha.is_finite() && f.is_finite())
        .collect();
    spectrum.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    if spectrum.is_empty() {
        return spectrum;
    }
    let mut unique: Vec<(f64, f64)> = Vec::with_capacity(spectrum.len());
    for (alpha, f) in spectrum {
        if let Some((last_alpha, last_f)) = unique.last_mut() {
            if (alpha - *last_alpha).abs() < 1e-12 {
                *last_f = (*last_f + f) * 0.5;
                continue;
            }
        }
        unique.push((alpha, f));
    }
    unique
}

fn interpolate_spectrum(spectrum: &[(f64, f64)], alpha: f64) -> Option<f64> {
    if spectrum.len() < 2 {
        return None;
    }
    if !alpha.is_finite() {
        return None;
    }
    let first = spectrum.first()?.0;
    let last = spectrum.last()?.0;
    if alpha < first || alpha > last {
        return None;
    }
    if (alpha - first).abs() < 1e-12 {
        return Some(spectrum.first()?.1);
    }
    if (alpha - last).abs() < 1e-12 {
        return Some(spectrum.last()?.1);
    }
    for pair in spectrum.windows(2) {
        let (a1, f1) = pair[0];
        let (a2, f2) = pair[1];
        if alpha >= a1 && alpha <= a2 {
            let denominator = a2 - a1;
            if denominator.abs() < 1e-12 {
                return Some((f1 + f2) * 0.5);
            }
            let weight = (alpha - a1) / denominator;
            let value = f1 + weight * (f2 - f1);
            if value.is_finite() {
                return Some(value);
            }
            return None;
        }
    }
    None
}

fn build_median_mfdfa_spectrum(spectra: &[Vec<(f64, f64)>]) -> Vec<SpectrumPoint> {
    let valid_spectra: Vec<&Vec<(f64, f64)>> = spectra
        .iter()
        .filter(|s| s.len() >= MIN_SPECTRUM_POINTS)
        .collect();
    if valid_spectra.len() < MIN_SPECTRUM_STOCKS {
        return Vec::new();
    }

    // Используем фиксированный диапазон α, а не min/max по всем тикерам
    let alpha_min = MFDFA_ALPHA_MIN;
    let alpha_max = MFDFA_ALPHA_MAX;

    let grid_size = MFDFA_GRID_POINTS.max(2);
    let mut result = Vec::with_capacity(grid_size);
    for i in 0..grid_size {
        let fraction = i as f64 / (grid_size - 1) as f64;
        let alpha = alpha_min + fraction * (alpha_max - alpha_min);
        let mut values = Vec::with_capacity(valid_spectra.len());
        for spectrum in &valid_spectra {
            if let Some(value) = interpolate_spectrum(spectrum.as_slice(), alpha) {
                // Отсекаем отрицательные f(α) — они не физичны
                if value.is_finite() && value >= 0.0 {
                    values.push(value);
                }
            }
        }
        // Требуем минимум тикеров в точке
        if values.len() < MFDFA_MIN_TICKERS_PER_POINT {
            continue;
        }
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median_value = quantile_sorted(&values, 0.50);
        let q25_value = quantile_sorted(&values, 0.25);
        let q75_value = quantile_sorted(&values, 0.75);
        let (Some(median_value), Some(q25_value), Some(q75_value)) = (median_value, q25_value, q75_value) else {
            continue;
        };
        if median_value.is_finite() && q25_value.is_finite() && q75_value.is_finite() {
            result.push(SpectrumPoint {
                alpha,
                median: median_value,
                q25: q25_value,
                q75: q75_value,
            });
        }
    }
    result
}

// ---------------------------------------------------------------------------
// AUC по вложенным моделям
// ---------------------------------------------------------------------------

fn plot_auc_by_block() -> Result<()> {
    let path = format!("{}/auc_summary.csv", OUTPUT_DIR);
    let content = std::fs::read_to_string(&path)?;
    let mut points: Vec<(String, f64, f64)> = Vec::new();
    for (i, line) in content.lines().enumerate() {
        if i == 0 { continue; } // заголовок
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() < 4 { continue; }
        let name = parts[0].to_string();
        let auc: f64 = parts[2].parse().unwrap_or(f64::NAN);
        let r2: f64 = parts[3].parse().unwrap_or(f64::NAN);
        if auc.is_finite() {
            points.push((name, auc, r2));
        }
    }
    if points.len() < 2 {
        return Ok(());
    }

    // Ось X — индекс модели, подписи сделаем вручную
    let n = points.len();
    let x_min = 0.0_f64;
    let x_max = (n - 1) as f64;

    let auc_vals: Vec<f64> = points.iter().map(|p| p.1).collect();
    let r2_vals: Vec<f64> = points.iter().map(|p| p.2).filter(|v| v.is_finite()).collect();

    // Общий диапазон по Y — чтобы AUC и R² поместились на одной оси
    let mut all_y: Vec<f64> = auc_vals.clone();
    all_y.extend(r2_vals.iter().copied());
    let (y_min, y_max) = match linear_range(&all_y) {
        Some(r) => r,
        None => return Ok(()),
    };

    let png_path = format!("{}/auc_by_block.png", OUTPUT_DIR);
    let svg_path = format!("{}/auc_by_block.svg", OUTPUT_DIR);

    let auc_points: Vec<(f64, f64)> = points
        .iter()
        .enumerate()
        .map(|(i, (_, auc, _))| (i as f64, *auc))
        .collect();
    let r2_points: Vec<(f64, f64)> = points
        .iter()
        .enumerate()
        .filter(|(_, (_, _, r2))| r2.is_finite())
        .map(|(i, (_, _, r2))| (i as f64, *r2))
        .collect();

    let labels: Vec<String> = points.iter().map(|p| p.0.clone()).collect();

        fn draw_auc_chart<DB: DrawingBackend>(
        root: &DrawingArea<DB, Shift>,
        x_min: f64,
        x_max: f64,
        y_min: f64,
        y_max: f64,
        n: usize,
        labels: &[String],
        auc_points: &[(f64, f64)],
        r2_points: &[(f64, f64)],
    ) -> Result<()>
    where
        DB::ErrorType: 'static,
    {
        let main_style = ShapeStyle::from(&BLACK).stroke_width(MAIN_LINE_WIDTH);
        let ref_style = ShapeStyle::from(&RED).stroke_width(IQR_LINE_WIDTH);

        root.fill(&WHITE)?;
        let mut chart = ChartBuilder::on(root)
            .caption(
                "AUC и pseudo-R² по блокам предикторов",
                FontDesc::new(FontFamily::SansSerif, FONT_SIZE, FontStyle::Normal),
            )
            .margin(40)
            .x_label_area_size(100)
            .y_label_area_size(140)
            .build_cartesian_2d(x_min..x_max, y_min..y_max)?;

        chart
            .configure_mesh()
            .x_desc("Модель")
            .y_desc("AUC / pseudo-R²")
            .label_style(FontDesc::new(FontFamily::SansSerif, 34.0, FontStyle::Normal))
            .axis_style(ShapeStyle::from(&BLACK).stroke_width(3))
            .x_labels(n)
            .x_label_formatter(&|x| {
                let idx = (*x).round() as isize;
                if idx < 0 || (idx as usize) >= labels.len() {
                    String::new()
                } else {
                    labels[idx as usize].clone()
                }
            })
            .draw()?;

        chart.draw_series(LineSeries::new(
            vec![(x_min, 0.5), (x_max, 0.5)],
            ref_style.clone(),
        ))?;

        chart
            .draw_series(LineSeries::new(auc_points.to_vec(), main_style.clone()))?
            .label("AUC")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], main_style.clone()));

        if !r2_points.is_empty() {
            chart
                .draw_series(LineSeries::new(r2_points.to_vec(), ref_style.clone()))?
                .label("pseudo-R²")
                .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], ref_style.clone()));
        }

        chart.draw_series(
            auc_points
                .iter()
                .map(|&(x, y)| Circle::new((x, y), MARKER_SIZE, ShapeStyle::from(&BLACK).filled())),
        )?;

        chart
            .configure_series_labels()
            .background_style(WHITE.mix(0.8))
            .border_style(BLACK)
            .label_font(FontDesc::new(FontFamily::SansSerif, 28.0, FontStyle::Normal))
            .draw()?;

        root.present()?;
        Ok(())
    }

    {
        let root = BitMapBackend::new(&png_path, (IMG_WIDTH, IMG_HEIGHT)).into_drawing_area();
        draw_auc_chart(
            &root, x_min, x_max, y_min, y_max, n, &labels, &auc_points, &r2_points,
        )?;
    }
    {
        let root = SVGBackend::new(&svg_path, (IMG_WIDTH, IMG_HEIGHT)).into_drawing_area();
        draw_auc_chart(
            &root, x_min, x_max, y_min, y_max, n, &labels, &auc_points, &r2_points,
        )?;
    }

    Ok(())
}


// ---------------------------------------------------------------------------
// Основная функция
// ---------------------------------------------------------------------------

pub fn generate_plots(selected_tickers: &[String]) -> Result<()> {
    let mut horizons_sorted: Vec<usize> = HORIZONS.to_vec();
    horizons_sorted.sort_unstable();
    horizons_sorted.dedup();
    if horizons_sorted.is_empty() {
        bail!("HORIZONS is empty");
    }
    if horizons_sorted.iter().any(|&h| h == 0) {
        bail!("HORIZONS contains zero");
    }

    let mut all_kurt_by_h: Vec<Vec<f64>> = vec![Vec::new(); horizons_sorted.len()];
    let mut all_alpha_by_h: Vec<Vec<f64>> = vec![Vec::new(); horizons_sorted.len()];
    let mut mfdfa_spectra: Vec<Vec<(f64, f64)>> = Vec::new();

    for ticker in selected_tickers {
        let bars = match load_bars_for_ticker(ticker) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("Warning: failed to load bars for {}: {}", ticker, e);
                continue;
            }
        };
        if bars.is_empty() {
            continue;
        }
        let closes: Vec<f64> = bars.iter().map(|b| b.close).collect();
        if closes.len() < 2 {
            continue;
        }

        for (index, &horizon) in horizons_sorted.iter().enumerate() {
            if closes.len() <= horizon {
                continue;
            }
            let returns: Vec<f64> = (0..closes.len().saturating_sub(horizon))
                .filter_map(|t| agg_log_return_from_prices(&closes, t, horizon))
                .filter(|r| r.is_finite())
                .collect();
            if returns.len() < MIN_RETURN_OBSERVATIONS {
                continue;
            }
            if let Some(k) = empirical_kurtosis(&returns) {
                if k.is_finite() && k > 0.0 {
                    all_kurt_by_h[index].push(k);
                }
            }
            let mut sorted_returns = returns;
            sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            if let Some(alpha) = tail_index(&sorted_returns) {
                if alpha.is_finite() && alpha > 0.0 {
                    all_alpha_by_h[index].push(alpha);
                }
            }
        }

        if !closes.iter().all(|price| price.is_finite() && *price > 0.0) {
            continue;
        }
        if closes.len() < MIN_MFDFA_OBSERVATIONS {
            continue;
        }
        let returns: Vec<f64> = closes
            .windows(2)
            .filter_map(|window| {
                let p0 = window[0];
                let p1 = window[1];
                if p0.is_finite() && p1.is_finite() && p0 > 0.0 && p1 > 0.0 {
                    let r = (p1 / p0).ln();
                    if r.is_finite() { Some(r) } else { None }
                } else {
                    None
                }
            })
            .collect();
        if returns.len() < MIN_MFDFA_OBSERVATIONS {
            continue;
        }
        let mf = mf_dfa(&returns, -5.0, 5.0, 0.5, None, 1);
        let spectrum = prepare_mfdfa_spectrum(&mf.alpha_q, &mf.f_alpha);
        if spectrum.len() >= MIN_SPECTRUM_POINTS {
            mfdfa_spectra.push(spectrum);
        }
    }

    let has_kurtosis = all_kurt_by_h.iter().any(|v| v.len() >= MIN_CROSS_SECTIONAL_N);
    let has_tail_index = all_alpha_by_h.iter().any(|v| v.len() >= MIN_CROSS_SECTIONAL_N);
    let has_mfdfa = mfdfa_spectra.len() >= MIN_SPECTRUM_STOCKS;

    if !has_kurtosis && !has_tail_index && !has_mfdfa {
        bail!("No valid data collected");
    }

    let main_style = ShapeStyle::from(&BLACK).stroke_width(MAIN_LINE_WIDTH);
    let iqr_style = ShapeStyle::from(&BLACK).stroke_width(IQR_LINE_WIDTH);

    // Рис. 1
    if has_kurtosis {
        let summaries: Vec<Option<SummaryStats>> = all_kurt_by_h.iter().map(|v| summary_stats(v)).collect();
        let median_points: Vec<(f64, f64)> = horizons_sorted.iter().zip(summaries.iter())
            .filter_map(|(&h, stats)| {
                let stats = (*stats)?;
                if stats.median.is_finite() && stats.median > 0.0 {
                    Some((h as f64, stats.median))
                } else {
                    None
                }
            })
            .collect();
        let q25_points: Vec<(f64, f64)> = horizons_sorted.iter().zip(summaries.iter())
            .filter_map(|(&h, stats)| {
                let stats = (*stats)?;
                if stats.q25.is_finite() && stats.q25 > 0.0 {
                    Some((h as f64, stats.q25))
                } else {
                    None
                }
            })
            .collect();
        let q75_points: Vec<(f64, f64)> = horizons_sorted.iter().zip(summaries.iter())
            .filter_map(|(&h, stats)| {
                let stats = (*stats)?;
                if stats.q75.is_finite() && stats.q75 > 0.0 {
                    Some((h as f64, stats.q75))
                } else {
                    None
                }
            })
            .collect();

        if median_points.len() >= 2 {
            let x_values: Vec<f64> = median_points.iter().map(|p| p.0).collect();
            let y_values: Vec<f64> = q25_points.iter().chain(q75_points.iter()).map(|p| p.1).collect();
            if let (Some((x_min, x_max)), Some((y_min, y_max))) = (positive_range(&x_values), positive_range(&y_values)) {
                let png_path = format!("{}/kurtosis.png", OUTPUT_DIR);
                let svg_path = format!("{}/kurtosis.svg", OUTPUT_DIR);
                // PNG
                {
                    let root = BitMapBackend::new(&png_path, (IMG_WIDTH, IMG_HEIGHT)).into_drawing_area();
                    root.fill(&WHITE)?;
                    let mut chart = ChartBuilder::on(&root)
                        .caption("Медианный эксцесс по рынку", FontDesc::new(FontFamily::SansSerif, FONT_SIZE, FontStyle::Normal))
                        .margin(40)
                        .x_label_area_size(110)
                        .y_label_area_size(140)
                        .build_cartesian_2d((x_min..x_max).log_scale(), (y_min..y_max).log_scale())?;
                    chart.configure_mesh()
                        .x_desc("Горизонт (мин)")
                        .y_desc("Эксцесс")
                        .label_style(FontDesc::new(FontFamily::SansSerif, 34.0, FontStyle::Normal))
                        .axis_style(ShapeStyle::from(&BLACK).stroke_width(3))
                        .draw()?;
                    chart.draw_series(LineSeries::new(q25_points.clone(), iqr_style.clone()))?;
                    chart.draw_series(LineSeries::new(q75_points.clone(), iqr_style.clone()))?;
                    chart.draw_series(LineSeries::new(median_points.clone(), main_style.clone()))?;
                    chart.draw_series(median_points.iter().map(|&(x, y)| Circle::new((x, y), MARKER_SIZE, ShapeStyle::from(&BLACK).filled())))?;
                    root.present()?;
                }
                // SVG
                {
                    let root = SVGBackend::new(&svg_path, (IMG_WIDTH, IMG_HEIGHT)).into_drawing_area();
                    root.fill(&WHITE)?;
                    let mut chart = ChartBuilder::on(&root)
                        .caption("Медианный эксцесс по рынку", FontDesc::new(FontFamily::SansSerif, FONT_SIZE, FontStyle::Normal))
                        .margin(40)
                        .x_label_area_size(110)
                        .y_label_area_size(140)
                        .build_cartesian_2d((x_min..x_max).log_scale(), (y_min..y_max).log_scale())?;
                    chart.configure_mesh()
                        .x_desc("Горизонт (мин)")
                        .y_desc("Эксцесс")
                        .label_style(FontDesc::new(FontFamily::SansSerif, 34.0, FontStyle::Normal))
                        .axis_style(ShapeStyle::from(&BLACK).stroke_width(3))
                        .draw()?;
                    chart.draw_series(LineSeries::new(q25_points, iqr_style.clone()))?;
                    chart.draw_series(LineSeries::new(q75_points, iqr_style.clone()))?;
                    chart.draw_series(LineSeries::new(median_points.clone(), main_style.clone()))?;
                    chart.draw_series(median_points.iter().map(|&(x, y)| Circle::new((x, y), MARKER_SIZE, ShapeStyle::from(&BLACK).filled())))?;
                    root.present()?;
                }
            }
        }
    }

    // Рис. 2
    if has_tail_index {
        let summaries: Vec<Option<SummaryStats>> = all_alpha_by_h.iter().map(|v| summary_stats(v)).collect();
        let median_points: Vec<(f64, f64)> = horizons_sorted.iter().zip(summaries.iter())
            .filter_map(|(&h, stats)| {
                let stats = (*stats)?;
                if stats.median.is_finite() && stats.median > 0.0 {
                    Some((h as f64, stats.median))
                } else {
                    None
                }
            })
            .collect();
        let q25_points: Vec<(f64, f64)> = horizons_sorted.iter().zip(summaries.iter())
            .filter_map(|(&h, stats)| {
                let stats = (*stats)?;
                if stats.q25.is_finite() && stats.q25 > 0.0 {
                    Some((h as f64, stats.q25))
                } else {
                    None
                }
            })
            .collect();
        let q75_points: Vec<(f64, f64)> = horizons_sorted.iter().zip(summaries.iter())
            .filter_map(|(&h, stats)| {
                let stats = (*stats)?;
                if stats.q75.is_finite() && stats.q75 > 0.0 {
                    Some((h as f64, stats.q75))
                } else {
                    None
                }
            })
            .collect();

        if median_points.len() >= 2 {
            let x_values: Vec<f64> = median_points.iter().map(|p| p.0).collect();
            let y_values: Vec<f64> = q25_points.iter().chain(q75_points.iter()).map(|p| p.1).collect();
            if let (Some((x_data_min, x_data_max)), Some((y_min, y_max))) = (linear_range(&x_values), positive_range(&y_values)) {
                let x_min = x_data_min.max(0.0);
                let x_max = x_data_max.max(x_min + 1.0);
                let png_path = format!("{}/tail_index.png", OUTPUT_DIR);
                let svg_path = format!("{}/tail_index.svg", OUTPUT_DIR);
                // PNG
                {
                    let root = BitMapBackend::new(&png_path, (IMG_WIDTH, IMG_HEIGHT)).into_drawing_area();
                    root.fill(&WHITE)?;
                    let mut chart = ChartBuilder::on(&root)
                        .caption("Медианный хвостовой индекс", FontDesc::new(FontFamily::SansSerif, FONT_SIZE, FontStyle::Normal))
                        .margin(40)
                        .x_label_area_size(110)
                        .y_label_area_size(140)
                        .build_cartesian_2d(x_min..x_max, y_min..y_max)?;
                    chart.configure_mesh()
                        .x_desc("Горизонт (мин)")
                        .y_desc("α")
                        .label_style(FontDesc::new(FontFamily::SansSerif, 34.0, FontStyle::Normal))
                        .axis_style(ShapeStyle::from(&BLACK).stroke_width(3))
                        .draw()?;
                    chart.draw_series(LineSeries::new(q25_points.clone(), iqr_style.clone()))?;
                    chart.draw_series(LineSeries::new(q75_points.clone(), iqr_style.clone()))?;
                    chart.draw_series(LineSeries::new(median_points.clone(), main_style.clone()))?;
                    chart.draw_series(median_points.iter().map(|&(x, y)| Circle::new((x, y), MARKER_SIZE, ShapeStyle::from(&BLACK).filled())))?;
                    root.present()?;
                }
                // SVG
                {
                    let root = SVGBackend::new(&svg_path, (IMG_WIDTH, IMG_HEIGHT)).into_drawing_area();
                    root.fill(&WHITE)?;
                    let mut chart = ChartBuilder::on(&root)
                        .caption("Медианный хвостовой индекс", FontDesc::new(FontFamily::SansSerif, FONT_SIZE, FontStyle::Normal))
                        .margin(40)
                        .x_label_area_size(110)
                        .y_label_area_size(140)
                        .build_cartesian_2d(x_min..x_max, y_min..y_max)?;
                    chart.configure_mesh()
                        .x_desc("Горизонт (мин)")
                        .y_desc("α")
                        .label_style(FontDesc::new(FontFamily::SansSerif, 34.0, FontStyle::Normal))
                        .axis_style(ShapeStyle::from(&BLACK).stroke_width(3))
                        .draw()?;
                    chart.draw_series(LineSeries::new(q25_points, iqr_style.clone()))?;
                    chart.draw_series(LineSeries::new(q75_points, iqr_style.clone()))?;
                    chart.draw_series(LineSeries::new(median_points.clone(), main_style.clone()))?;
                    chart.draw_series(median_points.iter().map(|&(x, y)| Circle::new((x, y), MARKER_SIZE, ShapeStyle::from(&BLACK).filled())))?;
                    root.present()?;
                }
            }
        }
    }

    // Рис. 4: AUC по блокам
    if let Err(e) = plot_auc_by_block() {
        eprintln!("Warning: AUC plot failed: {}", e);
    }

    // Рис. 3
    if has_mfdfa {
        let spectrum = build_median_mfdfa_spectrum(&mfdfa_spectra);
        if spectrum.len() >= 2 {
            let median_points: Vec<(f64, f64)> = spectrum.iter().map(|p| (p.alpha, p.median)).collect();
            let q25_points: Vec<(f64, f64)> = spectrum.iter().map(|p| (p.alpha, p.q25)).collect();
            let q75_points: Vec<(f64, f64)> = spectrum.iter().map(|p| (p.alpha, p.q75)).collect();
            let x_values: Vec<f64> = median_points.iter().map(|p| p.0).collect();
            let y_values: Vec<f64> = q25_points.iter().chain(q75_points.iter()).map(|p| p.1).collect();
            if let (Some((x_min, x_max)), Some((y_min, y_max))) = (linear_range(&x_values), linear_range(&y_values)) {
                let png_path = format!("{}/mfdfa.png", OUTPUT_DIR);
                let svg_path = format!("{}/mfdfa.svg", OUTPUT_DIR);
                // PNG
                {
                    let root = BitMapBackend::new(&png_path, (IMG_WIDTH, IMG_HEIGHT)).into_drawing_area();
                    root.fill(&WHITE)?;
                    let mut chart = ChartBuilder::on(&root)
                        .caption("Медианный мультифрактальный спектр", FontDesc::new(FontFamily::SansSerif, FONT_SIZE, FontStyle::Normal))
                        .margin(40)
                        .x_label_area_size(110)
                        .y_label_area_size(140)
                        .build_cartesian_2d(x_min..x_max, y_min..y_max)?;
                    chart.configure_mesh()
                        .x_desc("α")
                        .y_desc("f(α)")
                        .label_style(FontDesc::new(FontFamily::SansSerif, 34.0, FontStyle::Normal))
                        .axis_style(ShapeStyle::from(&BLACK).stroke_width(3))
                        .draw()?;
                    chart.draw_series(LineSeries::new(q25_points.clone(), iqr_style.clone()))?;
                    chart.draw_series(LineSeries::new(q75_points.clone(), iqr_style.clone()))?;
                    chart.draw_series(LineSeries::new(median_points.clone(), main_style.clone()))?;
                    let marker_count = 9usize.min(median_points.len());
                    if marker_count >= 2 {
                        let last = median_points.len() - 1;
                        let mut indices = Vec::with_capacity(marker_count);
                        for i in 0..marker_count {
                            indices.push(i * last / (marker_count - 1));
                        }
                        indices.dedup();
                        chart.draw_series(indices.into_iter().map(|index| {
                            let (x, y) = median_points[index];
                            Circle::new((x, y), MARKER_SIZE, ShapeStyle::from(&BLACK).filled())
                        }))?;
                    }
                    root.present()?;
                }
                // SVG
                {
                    let root = SVGBackend::new(&svg_path, (IMG_WIDTH, IMG_HEIGHT)).into_drawing_area();
                    root.fill(&WHITE)?;
                    let mut chart = ChartBuilder::on(&root)
                        .caption("Медианный мультифрактальный спектр", FontDesc::new(FontFamily::SansSerif, FONT_SIZE, FontStyle::Normal))
                        .margin(40)
                        .x_label_area_size(110)
                        .y_label_area_size(140)
                        .build_cartesian_2d(x_min..x_max, y_min..y_max)?;
                    chart.configure_mesh()
                        .x_desc("α")
                        .y_desc("f(α)")
                        .label_style(FontDesc::new(FontFamily::SansSerif, 34.0, FontStyle::Normal))
                        .axis_style(ShapeStyle::from(&BLACK).stroke_width(3))
                        .draw()?;
                    chart.draw_series(LineSeries::new(q25_points, iqr_style.clone()))?;
                    chart.draw_series(LineSeries::new(q75_points, iqr_style.clone()))?;
                    chart.draw_series(LineSeries::new(median_points.clone(), main_style.clone()))?;
                    let marker_count = 9usize.min(median_points.len());
                    if marker_count >= 2 {
                        let last = median_points.len() - 1;
                        let mut indices = Vec::with_capacity(marker_count);
                        for i in 0..marker_count {
                            indices.push(i * last / (marker_count - 1));
                        }
                        indices.dedup();
                        chart.draw_series(indices.into_iter().map(|index| {
                            let (x, y) = median_points[index];
                            Circle::new((x, y), MARKER_SIZE, ShapeStyle::from(&BLACK).filled())
                        }))?;
                    }
                    root.present()?;
                }
            }
        }
    }

    Ok(())
}