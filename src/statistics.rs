use rand::Rng;

// Вспомогательные статистические функции.
//
// Соглашение модуля: значения NaN считаются "отсутствующими" и никогда
// не участвуют в вычислениях. Бесконечности (`+inf`/`-inf`) трактуются
// как НЕконечные значения и тоже исключаются из `median`/`mean`,
// поскольку они искажают агрегаты.
// В `safe_sort` все NaN перемещаются в конец среза, оставляя
// отсортированными по возрастанию все валидные значения в начале.

/// Медиана массива, игнорируя NaN и бесконечности (`+inf`/`-inf`).
///
/// Возвращает `f64::NAN`, если после фильтрации не осталось значений.
pub fn median(data: &[f64]) -> f64 {
    let mut filtered: Vec<f64> = data.iter().copied().filter(|x| x.is_finite()).collect();

    if filtered.is_empty() {
        return f64::NAN;
    }

    // После фильтрации NaN отсутствуют, поэтому partial_cmp безопасен.
    filtered.sort_by(|a, b| a.partial_cmp(b).expect("filtered values are always finite"));

    let mid = filtered.len() / 2;
    if filtered.len() % 2 == 0 {
        (filtered[mid - 1] + filtered[mid]) / 2.0
    } else {
        filtered[mid]
    }
}

/// Среднее арифметическое, игнорируя NaN и бесконечности (`+inf`/`-inf`).
///
/// Возвращает `f64::NAN`, если после фильтрации не осталось значений.
pub fn mean(data: &[f64]) -> f64 {
    let mut sum = 0.0;
    let mut count: usize = 0;

    for &x in data {
        if x.is_finite() {
            sum += x;
            count += 1;
        }
    }

    if count == 0 {
        f64::NAN
    } else {
        sum / count as f64
    }
}

/// Безопасная сортировка по возрастанию с обработкой NaN.
///
/// Валидные значения сортируются обычным образом и располагаются
/// в начале среза; все NaN (в любом количестве) перемещаются в конец
/// в исходном относительном порядке между собой не гарантируется.
pub fn safe_sort(v: &mut [f64]) {
    v.sort_by(|a, b| match (a.is_nan(), b.is_nan()) {
        (true, true) => std::cmp::Ordering::Equal,
        (true, false) => std::cmp::Ordering::Greater, // NaN уходит в конец
        (false, true) => std::cmp::Ordering::Less,
        (false, false) => a.partial_cmp(b).expect("neither value is NaN here"),
    });
}


/// Bootstrap-оценка стандартной ошибки и 95% ДИ для медианы.
/// NaN и бесконечности игнорируются.
/// Возвращает (se, ci_low, ci_high).
pub fn bootstrap_median_ci(
    values: &[f64],
    b: usize,
    alpha: f64,
) -> (f64, f64, f64) {
    let clean: Vec<f64> = values
        .iter()
        .copied()
        .filter(|x| x.is_finite())
        .collect();
    let n = clean.len();
    if n < 3 {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    let mut rng = rand::thread_rng();
    let mut meds: Vec<f64> = Vec::with_capacity(b);
    let mut sample: Vec<f64> = Vec::with_capacity(n);
    for _ in 0..b {
        sample.clear();
        for _ in 0..n {
            let idx = rng.gen_range(0..n);
            sample.push(clean[idx]);
        }
        sample.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let m = if n % 2 == 0 {
            0.5 * (sample[n / 2 - 1] + sample[n / 2])
        } else {
            sample[n / 2]
        };
        meds.push(m);
    }
    meds.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean_m = meds.iter().sum::<f64>() / b as f64;
    let var = meds
        .iter()
        .map(|x| (x - mean_m).powi(2))
        .sum::<f64>()
        / (b as f64 - 1.0);
    let se = var.sqrt();
    let lo_idx = ((alpha / 2.0) * b as f64).floor() as usize;
    let hi_idx = ((1.0 - alpha / 2.0) * b as f64).ceil() as usize - 1;
    let ci_low = meds[lo_idx.min(b - 1)];
    let ci_high = meds[hi_idx.min(b - 1)];
    (se, ci_low, ci_high)
}

/// Односторонний bootstrap-тест H0: median >= null_value
/// против H1: median < null_value. NaN и бесконечности игнорируются.
/// Возвращает p-value.
pub fn bootstrap_median_less_than(
    values: &[f64],
    null_value: f64,
    b: usize,
) -> f64 {
    let clean: Vec<f64> = values
        .iter()
        .copied()
        .filter(|x| x.is_finite())
        .collect();
    let n = clean.len();
    if n < 3 {
        return f64::NAN;
    }

    // Наблюдаемая медиана
    let mut sorted = clean.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let obs_median = if n % 2 == 0 {
        0.5 * (sorted[n / 2 - 1] + sorted[n / 2])
    } else {
        sorted[n / 2]
    };

    // Сдвигаем данные так, чтобы под H0 их медиана была равна null_value
    let shift = null_value - obs_median;
    let shifted: Vec<f64> = clean.iter().map(|v| v + shift).collect();

    let mut rng = rand::thread_rng();
    let mut count = 0usize;
    let mut sample: Vec<f64> = Vec::with_capacity(n);
    for _ in 0..b {
        sample.clear();
        for _ in 0..n {
            let idx = rng.gen_range(0..n);
            sample.push(shifted[idx]);
        }
        sample.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let m = if n % 2 == 0 {
            0.5 * (sample[n / 2 - 1] + sample[n / 2])
        } else {
            sample[n / 2]
        };
        // Считаем бутстреп-медианы ниже наблюдаемой
        if m <= obs_median {
            count += 1;
        }
    }
    count as f64 / b as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn median_empty() {
        assert!(median(&[]).is_nan());
    }

    #[test]
    fn median_single() {
        assert_eq!(median(&[42.0]), 42.0);
    }

    #[test]
    fn median_odd() {
        assert_eq!(median(&[3.0, 1.0, 2.0]), 2.0);
    }

    #[test]
    fn median_even() {
        assert_eq!(median(&[1.0, 2.0, 3.0, 4.0]), 2.5);
    }

    #[test]
    fn median_ignores_nan_and_inf() {
        let data = [1.0, f64::NAN, 3.0, f64::INFINITY, 2.0, f64::NEG_INFINITY];
        assert_eq!(median(&data), 2.0);
    }

    #[test]
    fn median_all_invalid() {
        let data = [f64::NAN, f64::INFINITY, f64::NEG_INFINITY];
        assert!(median(&data).is_nan());
    }

    #[test]
    fn mean_empty() {
        assert!(mean(&[]).is_nan());
    }

    #[test]
    fn mean_basic() {
        assert_eq!(mean(&[1.0, 2.0, 3.0]), 2.0);
    }

    #[test]
    fn mean_ignores_nan_and_inf() {
        let data = [2.0, f64::NAN, 4.0, f64::INFINITY];
        assert_eq!(mean(&data), 3.0);
    }

    #[test]
    fn mean_all_invalid() {
        let data = [f64::NAN, f64::NEG_INFINITY];
        assert!(mean(&data).is_nan());
    }

    #[test]
    fn safe_sort_basic() {
        let mut v = [3.0, 1.0, 2.0];
        safe_sort(&mut v);
        assert_eq!(v, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn safe_sort_nan_goes_to_end() {
        let mut v = [f64::NAN, 3.0, f64::NAN, 1.0, 2.0];
        safe_sort(&mut v);
        assert_eq!(&v[..3], [1.0, 2.0, 3.0]);
        assert!(v[3].is_nan() && v[4].is_nan());
    }

    #[test]
    fn safe_sort_all_nan() {
        let mut v = [f64::NAN, f64::NAN, f64::NAN];
        safe_sort(&mut v);
        assert!(v.iter().all(|x| x.is_nan()));
    }

    #[test]
    fn safe_sort_with_infinities() {
        let mut v = [f64::INFINITY, 1.0, f64::NEG_INFINITY, f64::NAN, 0.0];
        safe_sort(&mut v);
        assert_eq!(&v[..4], [f64::NEG_INFINITY, 0.0, 1.0, f64::INFINITY]);
        assert!(v[4].is_nan());
    }

    #[test]
    fn safe_sort_empty() {
        let mut v: [f64; 0] = [];
        safe_sort(&mut v);
        assert!(v.is_empty());
    }

    #[test]
    fn bootstrap_median_ci_finite() {
        let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let (se, lo, hi) = bootstrap_median_ci(&data, 1000, 0.05);
        assert!(se.is_finite());
        assert!(lo <= hi);
    }

    #[test]
    fn bootstrap_median_ci_ignores_nan() {
        let mut data: Vec<f64> = (0..50).map(|i| i as f64).collect();
        data.push(f64::NAN);
        data.push(f64::INFINITY);
        let (se, _, _) = bootstrap_median_ci(&data, 500, 0.05);
        assert!(se.is_finite());
    }

    #[test]
    fn bootstrap_median_ci_too_short() {
        let (se, lo, hi) = bootstrap_median_ci(&[1.0, 2.0], 100, 0.05);
        assert!(se.is_nan() && lo.is_nan() && hi.is_nan());
    }

    #[test]
    fn bootstrap_less_than_ok() {
        // Данные с медианой около 0 — p-value должен быть около 0.5
        let data: Vec<f64> = (0..200).map(|i| (i as f64 - 100.0) / 100.0).collect();
        let p = bootstrap_median_less_than(&data, 0.0, 1000);
        assert!(p > 0.2 && p < 0.8);
    }

    #[test]
    fn bootstrap_less_than_shifted() {
        // Данные с медианой около -1 при null = 0 — p-value мал
        let data: Vec<f64> = (0..200).map(|i| (i as f64 - 300.0) / 100.0).collect();
        let p = bootstrap_median_less_than(&data, 0.0, 1000);
        assert!(p < 0.05);
    }

    #[test]
    fn bootstrap_less_than_ignores_nan() {
        let mut data: Vec<f64> = (0..100).map(|i| (i as f64 - 50.0) / 50.0).collect();
        data.push(f64::NAN);
        let p = bootstrap_median_less_than(&data, 0.0, 500);
        assert!(p.is_finite());
    }
}