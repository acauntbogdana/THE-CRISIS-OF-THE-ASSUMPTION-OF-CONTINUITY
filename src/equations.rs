use rand::{Rng, thread_rng};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use nalgebra::{DMatrix, DVector};
use nalgebra::linalg::SVD;

// ==================== Уровень 1 ====================
// Модуль для оценки хвостовых индексов и связанных величин
// на основе агрегированных логарифмических доходностей.
//
// Реализация следует обозначениям из статей:
// - Lieberman & Phillips (2022)
// - Liu & Chang (2023)
// - Einmahl & He (2023)

/// (1) Агрегированная логарифмическая доходность:
/// R_{t,h} = ln P_{t+h} - ln P_t
///
/// # Аргументы
/// * `prices` - вектор цен (непустой, положительные значения)
/// * `t` - индекс начального момента (0-based)
/// * `h` - горизонт агрегирования (число периодов)
///
/// # Возврат
/// `Some(R_{t,h})`, если вычисление возможно, иначе `None`
pub fn agg_log_return_from_prices(prices: &[f64], t: usize, h: usize) -> Option<f64> {
    if t + h >= prices.len() {
        return None;
    }

    let p0 = prices[t];
    let p1 = prices[t + h];

    if p0 <= 0.0 || p1 <= 0.0 {
        return None;
    }

    Some(p1.ln() - p0.ln())
}

/// (1) Если уже есть однопериодные лог-доходности r_t:
/// R_{t,h} = Σ_{k=0}^{h-1} r_{t+k}
///
/// # Аргументы
/// * `r` - вектор лог-доходностей (индексация с нуля: r[i] соответствует доходности за период (i, i+1))
/// * `t` - индекс начального момента (0-based)
/// * `h` - горизонт агрегирования
///
/// # Возврат
/// `Some(R_{t,h})`, если вычисление возможно.
/// Для `h = 0` возвращается `Some(0.0)`, что соответствует пустой сумме.
/// В прикладных задачах обычно требуют `h >= 1`.
pub fn agg_log_return_from_log_returns(r: &[f64], t: usize, h: usize) -> Option<f64> {
    // Необходимо, чтобы индекс t+h был валидным (т.е. t+h <= r.len())
    if t + h > r.len() {
        return None;
    }

    if h == 0 {
        return Some(0.0); // по определению пустой суммы
    }

    Some(r[t..t + h].iter().sum())
}

/// (3) Эмпирический эксцесс:
/// γ̂₂(h) = [ (1/N_h) Σ (R_{t,h} - R̄_h)^4 ] /
///          [ (1/N_h) Σ (R_{t,h} - R̄_h)^2 ]^2 - 3
///
/// Используется смещённая оценка (деление на n, не на n-1).
///
/// # Аргументы
/// * `returns` - срез агрегированных доходностей для данного горизонта
///
/// # Возврат
/// `Some(γ̂₂)`, если длина выборки ≥ 4 и дисперсия не нулевая.
pub fn empirical_kurtosis(returns: &[f64]) -> Option<f64> {
    if returns.len() < 4 {
        return None;
    }

    let n = returns.len() as f64;
    let mean = returns.iter().sum::<f64>() / n;

    let mut m2 = 0.0;
    let mut m4 = 0.0;

    for &x in returns {
        let dx = x - mean;
        m2 += dx * dx;
        m4 += dx * dx * dx * dx;
    }

    m2 /= n;
    m4 /= n;

    if m2.abs() < f64::EPSILON {
        return None;
    }

    Some(m4 / (m2 * m2) - 3.0)
}

/// (2), (4)
/// Теоретически: γ₂(R_{t,h}) ~ C h^{-β}
/// Оценка β из регрессии:
/// ln γ̂₂(h) = ln C - β ln h
///
/// # Аргументы
/// * `horizons` - вектор горизонтов h
/// * `kurtoses` - вектор соответствующих оценок эксцесса
///
/// # Возврат
/// `Some((β, C))`, если регрессия возможна.
///
/// # Замечания
/// Пары с неположительными горизонтами или неположительным эксцессом
/// отбрасываются (логарифм не определён). При отбрасывании выводится
/// предупреждение в stderr, чтобы исследователь мог учесть потерю данных.
pub fn estimate_beta(horizons: &[f64], kurtoses: &[f64]) -> Option<(f64, f64)> {
    if horizons.len() != kurtoses.len() {
        return None;
    }

    let mut dropped_nonpositive_h = 0;
    let mut dropped_nonpositive_kurt = 0;

    // Фильтруем только допустимые точки (h > 0, kurt > 0)
    let valid_pairs: Vec<(f64, f64)> = horizons
        .iter()
        .zip(kurtoses.iter())
        .filter(|&(h, &k)| {
            if *h <= 0.0 {
                dropped_nonpositive_h += 1;
                false
            } else if k <= 0.0 {
                dropped_nonpositive_kurt += 1;
                false
            } else {
                true
            }
        })
        .map(|(h, &k)| (*h, k))
        .collect();

    if dropped_nonpositive_h > 0 {
        eprintln!(
            "Предупреждение: отброшено {} точек с неположительным горизонтом h.",
            dropped_nonpositive_h
        );
    }
    if dropped_nonpositive_kurt > 0 {
        eprintln!(
            "Предупреждение: отброшено {} точек с неположительным эксцессом. \
             Рекомендуется проверить выборку или использовать робастные методы.",
            dropped_nonpositive_kurt
        );
    }

    if valid_pairs.len() < 2 {
        return None;
    }

    let n = valid_pairs.len() as f64;

    let mut sx = 0.0;
    let mut sy = 0.0;
    let mut sxx = 0.0;
    let mut sxy = 0.0;

    for (h, kurt) in valid_pairs {
        let lx = h.ln();
        let ly = kurt.ln();

        sx += lx;
        sy += ly;
        sxx += lx * lx;
        sxy += lx * ly;
    }

    let denom = n * sxx - sx * sx;

    if denom.abs() < 1e-15 {
        return None;
    }

    // slope = -β
    let slope = (n * sxy - sx * sy) / denom;

    // intercept = ln C
    let intercept = (sy - slope * sx) / n;

    // β = -slope, C = exp(intercept)
    Some((-slope, intercept.exp()))
}

/// (5) Оценка Хилла по верхним k порядковым статистикам.
///
/// На вход подаётся ряд, отсортированный по возрастанию:
/// X_{1:p} ≤ X_{2:p} ≤ ... ≤ X_{p:p}
///
/// # Аргументы
/// * `sorted_ascending` - отсортированный по возрастанию вектор положительных значений
/// * `k` - число экстремальных порядковых статистик для оценки (1 ≤ k < p)
///
/// # Возврат
/// `Some(γ̂)` — оценка параметра Хилла.
pub fn hill_estimator_with_k(sorted_ascending: &[f64], k: usize) -> Option<f64> {
    let p = sorted_ascending.len();

    if p < 2 {
        return None;
    }

    if k < 1 || k >= p {
        return None;
    }

    // X_{p-k:p} в 1-индексации соответствует sorted_ascending[p - k - 1]
    let base_index = p - k - 1;
    let x_base = sorted_ascending[base_index];

    if x_base <= 0.0 {
        return None;
    }

    let ln_base = x_base.ln();
    let mut sum = 0.0;

    for i in 0..k {
        // X_{p-i:p} в 1-индексации соответствует sorted_ascending[p - 1 - i]
        let x = sorted_ascending[p - 1 - i];

        if x <= 0.0 {
            return None;
        }

        sum += x.ln() - ln_base;
    }

    Some(sum / k as f64)
}

/// (5) Оценка Хилла с выбором k по умолчанию: k ≈ 0.05 * p.
///
/// # Аргументы
/// * `sorted_ascending` - отсортированный по возрастанию вектор положительных значений
///
/// # Возврат
/// `Some(γ̂)`, если оценка возможна.
pub fn hill_estimator(sorted_ascending: &[f64]) -> Option<f64> {
    let p = sorted_ascending.len();
    if p < 2 {
        return None;
    }
    let k = ((p as f64) * 0.05).round() as usize;
    hill_estimator_with_k(sorted_ascending, k)
}

/// (6) Показатель хвоста:
/// α̂_h = 1 / γ̂_h
///
/// # Аргументы
/// * `sorted_ascending` - отсортированный по возрастанию вектор положительных значений
///
/// # Возврат
/// `Some(α̂)`, если оценка Хилла успешна и не равна нулю.
pub fn tail_index(sorted_ascending: &[f64]) -> Option<f64> {
    let gamma = hill_estimator(sorted_ascending)?;

    if gamma.abs() < f64::EPSILON {
        return None;
    }

    Some(1.0 / gamma)
}

/// Асимптотическая стандартная ошибка Хилла.
///
/// Теорема 2.2:
/// √k (γ̂ - γ) → N(0, γ²(1 - R(1,1)))
///
/// Поэтому:
/// se = γ * sqrt(1 - R(1,1)) / sqrt(k)
///
/// # Аргументы
/// * `gamma` - оценка γ
/// * `r11` - значение R(1,1) (мера гетерогенности из статьи)
/// * `k` - использованное число порядковых статистик
///
/// # Возврат
/// `Some(se)`, если вычисление возможно.
pub fn hill_asymptotic_sd(gamma: f64, r11: f64, k: usize) -> Option<f64> {
    if k == 0 {
        return None;
    }

    let v = 1.0 - r11;

    if v < 0.0 {
        return None;
    }

    Some(gamma * v.sqrt() / (k as f64).sqrt())
}

/// (7) Проверка условия сглаживания:
/// α_h → ∞ при h → ∞
///
/// Функция возвращает α_h для каждого горизонта.
///
/// # Аргументы
/// * `sorted_by_h` - вектор отсортированных выборок для каждого горизонта.
///   Каждая выборка должна быть отсортирована по возрастанию.
///
/// # Возврат
/// Вектор `Option<f64>` той же длины, содержащий оценки α_h.
pub fn tail_indices_by_horizon(sorted_by_h: &[Vec<f64>]) -> Vec<Option<f64>> {
    sorted_by_h
        .iter()
        .map(|sample| tail_index(sample))
        .collect()
}

// ==================== Уровень 2 ====================

/// Выборочное среднее
pub fn mean(data: &[f64]) -> f64 {
    let n = data.len() as f64;
    data.iter().sum::<f64>() / n
}

/// Выборочная дисперсия (деление на n-1)
pub fn variance(data: &[f64]) -> f64 {
    let n = data.len() as f64;
    let m = mean(data);
    data.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1.0)
}

/// Автоковариация с лагом h (центрированная)
fn autocovariance(data: &[f64], h: usize) -> f64 {
    let n = data.len();
    if h >= n {
        return 0.0;
    }
    let m = mean(data);
    let mut sum = 0.0;
    for t in 0..(n - h) {
        sum += (data[t] - m) * (data[t + h] - m);
    }
    sum / n as f64
}

/// Долгосрочная дисперсия (LRV) с ядром Бартлетта
/// Возвращает s^2 (квадрат долгосрочного стандартного отклонения)
pub fn bartlett_lrv(data: &[f64], block_size: usize) -> f64 {
    let n = data.len();
    let m = block_size.min(n - 1); // окно не может быть больше n-1
    let mut lrv = 0.0;
    for h in 0..m {
        let weight = 1.0 - (h as f64) / (m as f64);
        let ac = if h == 0 {
            autocovariance(data, 0)
        } else {
            // симметричная сумма: gamma(h) + gamma(-h) = 2*gamma(h)
            2.0 * autocovariance(data, h)
        };
        lrv += weight * ac;
    }
    lrv
}

/// Статистика CUSUM для проверки стабильности среднего или дисперсии
/// data - исходный ряд (или квадраты остатков)
/// s - долгосрочное стандартное отклонение (sqrt LRV)
/// Возвращает статистику T (аналог T_hat_mu или T_hat_sigma)
pub fn cusum_statistic(data: &[f64], s: f64) -> f64 {
    let t = data.len() as f64;
    let s_total: f64 = data.iter().sum();
    let mut max_abs = 0.0f64;
    let mut s_partial = 0.0;
    for (i, &x) in data.iter().enumerate() {
        s_partial += x;
        let val = s_partial - ((i + 1) as f64 / t) * s_total;
        max_abs = max_abs.max(val.abs());
    }
    (1.0 / t.sqrt()) * (1.0 / s) * max_abs
}

/// Генерация бутстреп-выборки методом блочного wild bootstrap (BWB)
/// centered_data - исходные данные, центрированные вычитанием среднего
/// block_size - размер блока m
/// rng - генератор случайных чисел
pub fn bwb_sample(centered_data: &[f64], block_size: usize, rng: &mut impl Rng) -> Vec<f64> {
    let n = centered_data.len();
    let m = block_size;
    let num_blocks = (n as f64 / m as f64).ceil() as usize;
    let mut result = Vec::with_capacity(n);
    for i in 0..num_blocks {
        // знак Радемахера
        let sign = if rng.gen::<bool>() { 1.0 } else { -1.0 };
        let start = i * m;
        let end = (start + m).min(n);
        for j in start..end {
            result.push(sign * centered_data[j]);
        }
    }
    result.truncate(n);
    result
}

/// Адаптивный выбор размера блока m по формуле (3.1) из Lee & Baek (2020)
/// K - верхняя граница (обычно 100)
pub fn adaptive_block_size(data: &[f64], k: usize) -> usize {
    let n = data.len();
    if n < 2 {
        return 1;
    }
    // оценка коэффициента AR(1) методом наименьших квадратов
    let m = mean(data);
    let mut num = 0.0;
    let mut den = 0.0;
    for t in 1..n {
        num += (data[t] - m) * (data[t - 1] - m);
        den += (data[t - 1] - m).powi(2);
    }
    let rho = if den.abs() > 1e-10 { num / den } else { 0.0 };
    let rho = rho.clamp(-0.99, 0.99); // избегаем деления на ноль
    let val = 1.147 * (4.0 * (n as f64) * rho.powi(2) / ((1.0 - rho.powi(2)).powi(2))).powf(1.0 / 3.0);
    let m = (val.floor() as usize).min(k);
    m.max(1)
}

/// CUSUM-BWB тест для проверки гипотезы H0: среднее постоянно
/// Возвращает (статистика T_mu, p-value)
pub fn cusum_bwb_mean_test(data: &[f64], block_size: Option<usize>, b: usize) -> (f64, f64) {
    let n = data.len();
    assert!(n > 1, "Недостаточно данных");
    let m = block_size.unwrap_or_else(|| adaptive_block_size(data, 100));
    let m = m.min(n - 1);

    // исходная статистика
    let lrv = bartlett_lrv(data, m);
    let s_mu = lrv.sqrt().max(1e-12);
    let t_mu = cusum_statistic(data, s_mu);

    // центрированные данные для бутстрепа
    let x_mean = mean(data);
    let centered: Vec<f64> = data.iter().map(|x| x - x_mean).collect();

    let mut rng = thread_rng();
    let mut count_greater = 0usize;
    for _ in 0..b {
        let bwb = bwb_sample(&centered, m, &mut rng);
        let lrv_star = bartlett_lrv(&bwb, m);
        let s_star = lrv_star.sqrt().max(1e-12);
        let t_b = cusum_statistic(&bwb, s_star);
        if t_b > t_mu {
            count_greater += 1;
        }
    }
    let p_value = count_greater as f64 / b as f64;
    (t_mu, p_value)
}

/// Оценка точки разрыва среднего k_mu (формула 2.17)
fn estimate_mean_break(data: &[f64]) -> usize {
    let n = data.len();
    let s_total: f64 = data.iter().sum();
    let mut max_val = 0.0f64;
    let mut k_hat = 1usize;
    let mut s_partial = 0.0;
    for t in 0..n {
        s_partial += data[t];
        let val = (s_partial - ((t + 1) as f64 / n as f64) * s_total).abs();
        if val > max_val {
            max_val = val;
            k_hat = t + 1;
        }
    }
    k_hat
}

/// CUSUM-BWB тест для проверки гипотезы H0: дисперсия постоянна
/// Возвращает (статистика T_sigma, p-value)
pub fn cusum_bwb_variance_test(data: &[f64], block_size: Option<usize>, b: usize) -> (f64, f64) {
    let n = data.len();
    assert!(n > 1, "Недостаточно данных");

    // 1. Оценка точки разрыва среднего
    let k = estimate_mean_break(data);

    // 2. Вычитание среднего по сегментам, возведение в квадрат
    let mean1 = mean(&data[..k]);
    let mean2 = mean(&data[k..]);
    let mut y: Vec<f64> = Vec::with_capacity(n);
    for (i, &x) in data.iter().enumerate() {
        let r = if i < k { x - mean1 } else { x - mean2 };
        y.push(r * r);
    }

    let m = block_size.unwrap_or_else(|| adaptive_block_size(&y, 100));
    let m = m.min(n - 1);

    // 3. Исходная статистика
    let lrv = bartlett_lrv(&y, m);
    let s_sigma = lrv.sqrt().max(1e-12);
    let t_sigma = cusum_statistic(&y, s_sigma);

    // 4. Бутстреп на основе центрированных Y
    let y_mean = mean(&y);
    let centered_y: Vec<f64> = y.iter().map(|v| v - y_mean).collect();

    let mut rng = thread_rng();
    let mut count_greater = 0usize;
    for _ in 0..b {
        let bwb = bwb_sample(&centered_y, m, &mut rng);
        let lrv_star = bartlett_lrv(&bwb, m);
        let s_star = lrv_star.sqrt().max(1e-12);
        let t_b = cusum_statistic(&bwb, s_star);
        if t_b > t_sigma {
            count_greater += 1;
        }
    }
    let p_value = count_greater as f64 / b as f64;
    (t_sigma, p_value)
}

/// Результаты MF-DFA
pub struct MfDfaResult {
    pub q_values: Vec<f64>,
    pub h_q: Vec<f64>,      // обобщённый показатель Хёрста
    pub alpha_q: Vec<f64>,  // сингулярность
    pub f_alpha: Vec<f64>,  // спектр
    pub tau_q: Vec<f64>,    // масс функция
    pub delta_alpha: f64,   // ширина спектра сингулярности
    pub delta_h: f64,       // разброс h(q)
}

/// Линейная регрессия y = a + b*x, возвращает (a, b)
fn linear_regression(x: &[f64], y: &[f64]) -> (f64, f64) {
    let n = x.len() as f64;
    let x_mean = x.iter().sum::<f64>() / n;
    let y_mean = y.iter().sum::<f64>() / n;
    let mut num = 0.0;
    let mut den = 0.0;
    for i in 0..x.len() {
        num += (x[i] - x_mean) * (y[i] - y_mean);
        den += (x[i] - x_mean).powi(2);
    }
    let b = num / den;
    let a = y_mean - b * x_mean;
    (a, b)
}

/// Численная производная (центральная разность)
fn numerical_derivative(x: &[f64], y: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mut dy = vec![0.0; n];
    if n < 3 {
        return dy;
    }
    for i in 0..n {
        if i == 0 {
            dy[i] = (y[i + 1] - y[i]) / (x[i + 1] - x[i]);
        } else if i == n - 1 {
            dy[i] = (y[i] - y[i - 1]) / (x[i] - x[i - 1]);
        } else {
            dy[i] = (y[i + 1] - y[i - 1]) / (x[i + 1] - x[i - 1]);
        }
    }
    dy
}

/// Выполнение MF-DFA анализа
/// data - временной ряд (обычно доходности)
/// q_min, q_max, q_step - диапазон параметра q
/// scales - опциональный набор масштабов s; если None, генерируется логарифмически
/// poly_degree - степень полинома для удаления тренда (обычно 1)
pub fn mf_dfa(
    data: &[f64],
    q_min: f64,
    q_max: f64,
    q_step: f64,
    scales: Option<Vec<usize>>,
    poly_degree: usize,
) -> MfDfaResult {
    let n = data.len();
    assert!(n > 10, "Ряд слишком короткий для MF-DFA");

    // Профиль X_t
    let x_mean = mean(data);
    let mut profile = Vec::with_capacity(n);
    let mut cum = 0.0;
    for &x in data {
        cum += x - x_mean;
        profile.push(cum);
    }

    // Определяем масштабы s
    let s_values: Vec<usize> = match scales {
        Some(v) => v,
        None => {
            let s_min = 10usize;
            let s_max = (n / 4).max(s_min + 1);
            let num_scales = 30;
            let log_min = (s_min as f64).ln();
            let log_max = (s_max as f64).ln();
            (0..num_scales)
                .map(|i| {
                    let frac = i as f64 / (num_scales - 1) as f64;
                    let log_s = log_min + frac * (log_max - log_min);
                    log_s.exp().round() as usize
                })
                .collect::<Vec<_>>()
                .into_iter()
                .filter(|&s| s >= s_min && s <= s_max && s < n)
                .collect::<Vec<_>>()
        }
    };
    if s_values.is_empty() {
        panic!("Нет допустимых масштабов s");
    }

    // Генерируем значения q
    let mut q_values = Vec::new();
    let mut q = q_min;
    while q <= q_max + 1e-12 {
        q_values.push(q);
        q += q_step;
    }

    // Для каждого q храним log(F_q(s)) для регрессии
    let mut log_f_by_q: Vec<Vec<f64>> = vec![Vec::with_capacity(s_values.len()); q_values.len()];

    // Цикл по масштабам
    for &s in &s_values {
        let ns = n / s; // число сегментов в одном направлении
        if ns < 2 {
            continue;
        }
        let total_segments = 2 * ns;
        let mut f2_values = Vec::with_capacity(total_segments);

        // Прямой проход
        for nu in 0..ns {
            let start = nu * s;
            let end = start + s;
            let segment = &profile[start..end];
            // Подгонка полинома степени poly_degree
            let xs: Vec<f64> = (0..s).map(|j| (j + 1) as f64).collect();
            let trend = fit_polynomial(&xs, segment, poly_degree);
            let mut sum_sq = 0.0;
            for j in 0..s {
                let resid = segment[j] - trend[j];
                sum_sq += resid * resid;
            }
            f2_values.push(sum_sq / s as f64);
        }

        // Обратный проход (с конца)
        for nu in 0..ns {
            let start = n - (nu + 1) * s;
            let end = start + s;
            let segment = &profile[start..end];
            let xs: Vec<f64> = (0..s).map(|j| (j + 1) as f64).collect();
            let trend = fit_polynomial(&xs, segment, poly_degree);
            let mut sum_sq = 0.0;
            for j in 0..s {
                let resid = segment[j] - trend[j];
                sum_sq += resid * resid;
            }
            f2_values.push(sum_sq / s as f64);
        }

        // Вычисляем F_q(s) для всех q
        for (qi, &q_val) in q_values.iter().enumerate() {
            let fq = if q_val.abs() < 1e-12 {
                // q = 0
                let sum_ln: f64 = f2_values.iter().map(|v| v.ln()).sum();
                (sum_ln / (4.0 * ns as f64)).exp()
            } else {
                let sum_pow: f64 = f2_values.iter().map(|v| v.powf(q_val / 2.0)).sum();
                (sum_pow / (total_segments as f64)).powf(1.0 / q_val)
            };
            log_f_by_q[qi].push(fq.ln());
        }
    }

    // Для каждого q выполняем линейную регрессию log(F_q(s)) ~ log(s)
    let mut h_q = Vec::with_capacity(q_values.len());
    let log_s: Vec<f64> = s_values.iter().map(|&s| (s as f64).ln()).collect();

    for (_qi, log_f) in log_f_by_q.iter().enumerate() {
        if log_f.len() < 2 {
            h_q.push(f64::NAN);
            continue;
        }
        let (_, slope) = linear_regression(&log_s, log_f);
        h_q.push(slope);
    }

    // Производная h'(q)
    let h_deriv = numerical_derivative(&q_values, &h_q);

    // Вычисляем α(q), f(α), τ(q)
    let mut alpha_q = Vec::with_capacity(q_values.len());
    let mut f_alpha = Vec::with_capacity(q_values.len());
    let mut tau_q = Vec::with_capacity(q_values.len());

    for i in 0..q_values.len() {
        let q = q_values[i];
        let h = h_q[i];
        let hd = h_deriv[i];
        let alpha = if hd.is_finite() { h + q * hd } else { f64::NAN };
        let f = if alpha.is_finite() { q * (alpha - h) + 1.0 } else { f64::NAN };
        let tau = q * h - 1.0;
        alpha_q.push(alpha);
        f_alpha.push(f);
        tau_q.push(tau);
    }

    let valid_alpha: Vec<f64> = alpha_q.iter().cloned().filter(|v| v.is_finite()).collect();
    let valid_h: Vec<f64> = h_q.iter().cloned().filter(|v| v.is_finite()).collect();
    let delta_alpha = if valid_alpha.len() >= 2 {
        let max_a = valid_alpha.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min_a = valid_alpha.iter().cloned().fold(f64::INFINITY, f64::min);
        max_a - min_a
    } else {
        f64::NAN
    };
    let delta_h = if valid_h.len() >= 2 {
        let max_h = valid_h.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min_h = valid_h.iter().cloned().fold(f64::INFINITY, f64::min);
        max_h - min_h
    } else {
        f64::NAN
    };

    MfDfaResult {
        q_values,
        h_q,
        alpha_q,
        f_alpha,
        tau_q,
        delta_alpha,
        delta_h,
    }
}

/// Подгонка полинома заданной степени к данным (x, y) и возврат предсказанных значений
fn fit_polynomial(x: &[f64], y: &[f64], degree: usize) -> Vec<f64> {
    let n = x.len();
    // Используем метод наименьших квадратов через нормальные уравнения
    let mut matrix = vec![vec![0.0; degree + 1]; degree + 1];
    let mut rhs = vec![0.0; degree + 1];
    for i in 0..=degree {
        for j in 0..=degree {
            let mut sum = 0.0;
            for k in 0..n {
                sum += x[k].powi(i as i32) * x[k].powi(j as i32);
            }
            matrix[i][j] = sum;
        }
        let mut sum = 0.0;
        for k in 0..n {
            sum += x[k].powi(i as i32) * y[k];
        }
        rhs[i] = sum;
    }
    // Решение системы линейных уравнений (метод Гаусса)
    let coeffs = solve_linear_system(&matrix, &rhs);
    // Вычисляем предсказанные значения
    (0..n)
        .map(|k| {
            let mut pred = 0.0;
            for (i, &c) in coeffs.iter().enumerate() {
                pred += c * x[k].powi(i as i32);
            }
            pred
        })
        .collect()
}

/// Решение системы линейных уравнений Ax = b методом Гаусса с частичным выбором ведущего элемента
fn solve_linear_system(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = a.len();
    let mut m = a.to_vec();
    let mut rhs = b.to_vec();
    for col in 0..n {
        // Поиск максимального элемента в столбце
        let mut max_row = col;
        for row in (col + 1)..n {
            if m[row][col].abs() > m[max_row][col].abs() {
                max_row = row;
            }
        }
        if m[max_row][col].abs() < 1e-12 {
            panic!("Матрица вырождена");
        }
        // Обмен строк
        m.swap(col, max_row);
        rhs.swap(col, max_row);
        // Нормализация и исключение
        let pivot = m[col][col];
        for j in col..n {
            m[col][j] /= pivot;
        }
        rhs[col] /= pivot;
        for row in 0..n {
            if row != col && m[row][col].abs() > 1e-12 {
                let factor = m[row][col];
                for j in col..n {
                    m[row][j] -= factor * m[col][j];
                }
                rhs[row] -= factor * rhs[col];
            }
        }
    }
    rhs
}

/// Результат проверки эргодичности через скорость убывания дисперсии
pub struct VarianceDecayResult {
    pub m_values: Vec<usize>,
    pub ratio: Vec<f64>, // Var(r̄_m) * m / σ²
}

/// Проверка эргодичности: Var(r̄_m) ~ σ²/m
/// data - исходный ряд
/// max_m - максимальный размер блока для проверки
/// b - число бутстреп-репликаций для оценки дисперсии среднего блока
pub fn variance_decay_check(data: &[f64], max_m: usize, b: usize) -> VarianceDecayResult {
    let n = data.len();
    assert!(n > max_m, "max_m должно быть меньше длины ряда");

    let sigma2 = variance(data);
    let mut rng = thread_rng();

    let mut m_values = Vec::new();
    let mut ratios = Vec::new();

    for m in 1..=max_m {
        // Оценка Var(r̄_m) через случайные блоки длины m
        let mut block_means = Vec::with_capacity(b);
        for _ in 0..b {
            let start = rng.gen_range(0..=(n - m));
            let block = &data[start..start + m];
            block_means.push(mean(block));
        }
        let var_block_mean = variance(&block_means);
        let ratio = var_block_mean * m as f64 / sigma2;
        m_values.push(m);
        ratios.push(ratio);
    }

    VarianceDecayResult {
        m_values,
        ratio: ratios,
    }
}

// ==================== Уровень 3 ====================
// Реализация оценки интегрированной ковариации Truncated S-TSRV
// согласно статье Дачуана Чэня (уровень 3) с исправлениями.

#[derive(Debug, Clone)]
pub struct StsrvParams {
    pub k: usize,
    pub j: usize,
    pub m_preavg: usize,
    pub threshold: f64,
    pub max_tau_step: f64,
    pub n_blocks: usize,
}

impl StsrvParams {
    pub fn new(k: usize, j: usize, m_preavg: usize, threshold: f64, max_tau_step: f64, n_blocks: usize) -> Self {
        assert!(k > j, "K должно быть больше J");
        assert!(m_preavg > 0, "m_preavg должен быть положительным");
        assert!(threshold > 0.0, "threshold должен быть положительным");
        assert!(max_tau_step > 0.0, "max_tau_step должен быть положительным");
        assert!(n_blocks > 0, "n_blocks должен быть положительным");
        assert!(k + j <= n_blocks, "b = K+J не должно превышать N");
        Self { k, j, m_preavg, threshold, max_tau_step, n_blocks }
    }

    pub fn boundary_correction(&self) -> f64 {
        1.0 - (self.k + self.j) as f64 / self.n_blocks as f64
    }

    pub fn scale_difference(&self) -> usize {
        self.k - self.j
    }

    pub fn convergence_rate(&self) -> f64 {
        ((self.scale_difference() as f64) * self.max_tau_step).sqrt()
    }

    /// Параметр ξ из формулы (12):
    /// ξ = N * (M_n^-)^2 / ((K-J)^3 * Δτ)
    pub fn xi(&self) -> f64 {
        let k_minus_j = self.scale_difference() as f64;
        let m = self.m_preavg as f64;
        self.n_blocks as f64 * m.powi(2) / (k_minus_j.powi(3) * self.max_tau_step)
    }
}

/// Предварительное усреднение и усечение приращений.
/// Возвращает ряд, в котором приращения, превышающие порог, обнулены.
/// Это эквивалентно замене текущего значения на предыдущее.
pub fn preaverage_and_truncate(prices: &[f64], params: &StsrvParams) -> Vec<f64> {
    let m = params.m_preavg;
    let n = prices.len();
    assert!(n >= m, "Недостаточно цен для окна предварительного усреднения");

    // Шаг 1: скользящее среднее
    let mut smoothed: Vec<f64> = Vec::with_capacity(n - m + 1);
    for i in 0..=(n - m) {
        let avg: f64 = prices[i..i + m].iter().sum::<f64>() / m as f64;
        smoothed.push(avg);
    }

    // Шаг 2: усечение приращений усреднённого ряда
    if smoothed.len() <= 1 {
        return smoothed;
    }
    let mut truncated = Vec::with_capacity(smoothed.len());
    truncated.push(smoothed[0]);
    for i in 1..smoothed.len() {
        let delta = smoothed[i] - smoothed[i - 1];
        if delta.abs() <= params.threshold {
            // Приращение в пределах порога – сохраняем значение
            truncated.push(smoothed[i]);
        } else {
            // Приращение превышает порог – обнуляем его,
            // поэтому оставляем значение равным предыдущему.
            truncated.push(truncated[i - 1]);
        }
    }
    truncated
}

/// Вычисление оценки масштаба для заданного лага `scale` (K или J)
/// с учётом граничных весов 1/2 (формула (10) из статьи).
pub fn compute_scale_estimate(
    y_r: &[f64],
    y_s: &[f64],
    scale: usize,
    params: &StsrvParams,
) -> f64 {
    let n = y_r.len().min(y_s.len());
    let k = scale;
    let b = params.k + params.j;
    if n <= k || b > n {
        return 0.0;
    }
    let n_star = n;
    let last_idx = n_star - k - 1;

    let s1_start = 0usize;
    let s1_end = (b - k).saturating_sub(1).min(last_idx);

    let s2_start = (b - k).max(0);
    let s2_end = (n_star - b - 1).min(last_idx);

    let s3_start = n_star - b;
    let s3_end = last_idx;

    let mut sum = 0.0;
    for i in s1_start..=s1_end {
        let dy_r = y_r[i + k] - y_r[i];
        let dy_s = y_s[i + k] - y_s[i];
        sum += 0.5 * dy_r * dy_s;
    }
    for i in s2_start..=s2_end {
        let dy_r = y_r[i + k] - y_r[i];
        let dy_s = y_s[i + k] - y_s[i];
        sum += dy_r * dy_s;
    }
    for i in s3_start..=s3_end {
        let dy_r = y_r[i + k] - y_r[i];
        let dy_s = y_s[i + k] - y_s[i];
        sum += 0.5 * dy_r * dy_s;
    }
    sum
}

/// Truncated S-TSRV оценка интегрированной ковариации (формулы (9) и (10)).
pub fn truncated_stsrv(
    y_r: &[f64],
    y_s: &[f64],
    params: &StsrvParams,
) -> f64 {
    let k_scale = compute_scale_estimate(y_r, y_s, params.k, params);
    let j_scale = compute_scale_estimate(y_r, y_s, params.j, params);
    let denominator = params.boundary_correction() * (params.scale_difference() as f64);
    (params.k as f64 * k_scale - params.j as f64 * j_scale) / denominator
}

/// Подынтегральное выражение для асимптотической дисперсии (формула (11))
/// с симметризацией [2][2]. Использует исправленный параметр ξ.
pub fn asymptotic_variance_integrand(
    cov_at_u: &dyn Fn(usize, usize) -> f64,
    noise_at_u: &dyn Fn(usize, usize) -> f64,
    r1: usize, s1: usize, r2: usize, s2: usize,
    params: &StsrvParams,
    total_time: f64,
) -> f64 {
    let c = |a: usize, b: usize| cov_at_u(a, b);
    let n = |a: usize, b: usize| noise_at_u(a, b);

    let signal_sym = c(r1, r2) * c(s1, s2)
        + c(r1, s2) * c(s1, r2)
        + c(r2, s1) * c(s2, r1)
        + c(r2, s2) * c(s1, r1);
    let signal_part = (1.0 / 3.0) * signal_sym;

    let noise_sym = n(r1, r2) * n(s1, s2)
        + n(r1, s2) * n(s1, r2)
        + n(r2, s1) * n(s2, r1)
        + n(r2, s2) * n(s1, r1);
    let noise_part = (2.0 * params.xi() / total_time) * noise_sym;

    signal_part + noise_part
}

// ==================== Уровень 4 ====================

/// Логистическая функция
fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}

/// Вычисление линейного предиктора η = X β
fn linear_predictor(x: ArrayView1<f64>, beta: ArrayView1<f64>) -> f64 {
    x.dot(&beta)
}

/// Логарифмическое правдоподобие для логистической регрессии
/// X: матрица предикторов (n x p), первая колонка должна быть единицами для константы
/// y: вектор бинарных исходов (0/1)
/// beta: вектор коэффициентов
fn log_likelihood(x: ArrayView2<f64>, y: ArrayView1<f64>, beta: ArrayView1<f64>) -> f64 {
    let n = x.nrows();
    let mut ll = 0.0;
    let eps = 1e-12; // защита от логарифма нуля
    for i in 0..n {
        let eta = linear_predictor(x.row(i), beta);
        let p = sigmoid(eta).clamp(eps, 1.0 - eps);
        ll += y[i] * p.ln() + (1.0 - y[i]) * (1.0 - p).ln();
    }
    ll
}

/// Гессиан логарифмического правдоподобия (отрицательный, т.к. выпуклая функция)
fn hessian(x: ArrayView2<f64>, beta: ArrayView1<f64>) -> Array2<f64> {
    let n = x.nrows();
    let p = x.ncols();
    let mut h = Array2::<f64>::zeros((p, p));
    for i in 0..n {
        let eta = linear_predictor(x.row(i), beta);
        let p = sigmoid(eta);
        let w = p * (1.0 - p);
        let xi = x.row(i).to_owned().insert_axis(Axis(0)); // 1 x p
        h = h + &(xi.t().dot(&xi) * w);
    }
    h
}

/// Псевдообращение матрицы через SVD с отсечением малых сингулярных чисел
pub fn pseudo_inverse(a: ArrayView2<f64>) -> Array2<f64> {
    let n = a.nrows();
    let m = a.ncols();
    let matrix = DMatrix::from_row_slice(n, m, a.as_slice().unwrap());
    let svd = SVD::new(matrix, true, true);
    let eps = 1e-12;
    let mut singular_inv = DMatrix::zeros(m, n);
    for i in 0..svd.singular_values.len().min(n).min(m) {
        let val = svd.singular_values[i];
        if val > eps {
            singular_inv[(i, i)] = 1.0 / val;
        }
    }
    let result = svd.v_t.as_ref().unwrap().transpose()
        * &singular_inv
        * svd.u.as_ref().unwrap().transpose();
    Array2::from_shape_vec((m, n), result.iter().copied().collect()).unwrap()
}

/// Подгонка логистической регрессии методом IRLS (iteratively reweighted least squares)
/// Возвращает (beta, Hessian, scores)
pub fn fit_logistic(x: ArrayView2<f64>, y: ArrayView1<f64>) -> (Array1<f64>, Array2<f64>, Array2<f64>) {
    let n = x.nrows();
    let p = x.ncols();
    let mut beta = Array1::<f64>::zeros(p);

    let max_iter = 100;
    let tol = 1e-8;
    let eps = 1e-12; // для стабилизации весов

    for _ in 0..max_iter {
        let mut z = Array1::<f64>::zeros(n);
        let mut w = Array1::<f64>::zeros(n);
        for i in 0..n {
            let eta = linear_predictor(x.row(i), beta.view());
            let p = sigmoid(eta);
            w[i] = (p * (1.0 - p)).max(eps); // избегаем нулевых весов
            z[i] = eta + (y[i] - p) / w[i];
        }
        let w_sqrt = w.mapv(f64::sqrt);
        let mut x_weighted = x.to_owned();
        for i in 0..n {
            let mut row = x_weighted.row_mut(i);
            row.mapv_inplace(|v| v * w_sqrt[i]);
        }
        let z_weighted = z * &w_sqrt;
        let x_matrix = DMatrix::from_row_slice(n, p, x_weighted.as_slice().unwrap());
        let z_vector = DVector::from_row_slice(z_weighted.as_slice().unwrap());
        let svd = SVD::new(x_matrix, true, true);
        let beta_new_vector = svd.solve(&z_vector, 1e-12)
            .expect("SVD solve failed");
        let beta_new = Array1::from_shape_vec(p, beta_new_vector.iter().copied().collect()).unwrap();
        let diff = (&beta_new - &beta).mapv(f64::abs).sum();
        beta = beta_new;
        if diff < tol {
            break;
        }
    }

        let h = hessian(x, beta.view());

        // Матрица индивидуальных скоров n×p: s_i = x_i * (y_i - p_i)
        let mut scores = Array2::<f64>::zeros((n, p));
        for i in 0..n {
            let eta = linear_predictor(x.row(i), beta.view());
            let p_i = sigmoid(eta);
            let residual = y[i] - p_i;
            let row = x.row(i).to_owned() * residual;
            scores.row_mut(i).assign(&row);
    }

    (beta, h, scores)
}

/// Предсказанные вероятности P(y=1|x).
pub fn predict_proba(x: ArrayView2<f64>, beta: ArrayView1<f64>) -> Array1<f64> {
    let n = x.nrows();
    let mut p = Array1::<f64>::zeros(n);
    for i in 0..n {
        p[i] = sigmoid(linear_predictor(x.row(i), beta));
    }
    p
}

/// Псевдо-R² МакФаддена: 1 - LL_model / LL_null.
pub fn pseudo_r2_mcfadden(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    beta: ArrayView1<f64>,
) -> f64 {
    let n = y.len() as f64;
    let ll_model = log_likelihood(x, y, beta);
    let p_bar = (y.sum() / n).clamp(1e-12, 1.0 - 1e-12);
    let ll_null = n * (p_bar * p_bar.ln() + (1.0 - p_bar) * (1.0 - p_bar).ln());
    if ll_null.abs() < 1e-12 {
        return f64::NAN;
    }
    1.0 - ll_model / ll_null
}

/// AUC (площадь под ROC-кривой) через статистику Манна–Уитни.
pub fn auc_score(y: ArrayView1<f64>, p: ArrayView1<f64>) -> f64 {
    let n = y.len();
    let n_pos = y.iter().filter(|&&v| v > 0.5).count() as f64;
    let n_neg = n as f64 - n_pos;
    if n_pos == 0.0 || n_neg == 0.0 {
        return f64::NAN;
    }
    let mut pairs: Vec<(f64, f64)> = (0..n).map(|i| (p[i], y[i])).collect();
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap()); // по убыванию вероятности
    let mut tp = 0.0;
    let mut auc = 0.0;
    for (_, yi) in &pairs {
        if *yi > 0.5 {
            tp += 1.0;
        } else {
            auc += tp;
        }
    }
    auc / (n_pos * n_neg)
}

/// Кластеризованная ковариационная матрица (sandwich) для одного типа кластеризации
pub fn cluster_cov_oneway(
    scores: ArrayView2<f64>,
    cluster_ids: &[usize],
    hessian: ArrayView2<f64>,
) -> Array2<f64> {
    let n = scores.nrows();
    let p = scores.ncols();

    let mut clusters: Vec<usize> = cluster_ids.to_vec();
    clusters.sort_unstable();
    clusters.dedup();

    let mut q = Array2::<f64>::zeros((p, p));
    for &cluster in &clusters {
        let mut s_cluster = Array1::<f64>::zeros(p);
        for i in 0..n {
            if cluster_ids[i] == cluster {
                s_cluster = s_cluster + &scores.row(i);
            }
        }
        q = q + &s_cluster.clone().insert_axis(Axis(0)).t().dot(&s_cluster.insert_axis(Axis(0)));
    }

    // Используем псевдообращение вместо прямого inv()
    let h_inv = pseudo_inverse(hessian);
    h_inv.dot(&q).dot(&h_inv)
}

/// Мультикластерная ковариационная матрица для двух измерений (бумага и дата)
pub fn cluster_cov_twoway(
    paper_ids: &[usize],
    date_ids: &[usize],
    scores: ArrayView2<f64>,
    hessian: ArrayView2<f64>,
) -> Array2<f64> {
    let v_paper = cluster_cov_oneway(scores, paper_ids, hessian);
    let v_date = cluster_cov_oneway(scores, date_ids, hessian);

    // Пересечение кластеров: каждая пара (paper, date) уникальна,
    // поэтому используем единичную кластеризацию (HC0)
    let intersect_ids: Vec<usize> = (0..scores.nrows()).collect();
    let v_intersect = cluster_cov_oneway(scores, &intersect_ids, hessian);

    v_paper + v_date - v_intersect
}

// ==================== Уровень 5 ====================
pub fn economic_filter_phi(r_th: f64, spread_t: f64) -> Option<f64> {
    if spread_t <= 0.0 {
        None
    } else {
        Some(r_th.abs() / spread_t)
    }
}

// ==================== Тесты ====================
#[cfg(test)]
mod tests {
    use super::*;

    // Тесты уровня 1
    #[test]
    fn test_agg_log_return_from_prices() {
        let prices = vec![100.0, 101.0, 102.0];
        assert_eq!(agg_log_return_from_prices(&prices, 0, 1), Some((101.0f64).ln() - (100.0f64).ln()));
        assert_eq!(agg_log_return_from_prices(&prices, 0, 2), Some((102.0f64).ln() - (100.0f64).ln()));
        assert_eq!(agg_log_return_from_prices(&prices, 1, 1), Some((102.0f64).ln() - (101.0f64).ln()));
        assert_eq!(agg_log_return_from_prices(&prices, 1, 2), None);
    }

    #[test]
    fn test_agg_log_return_from_log_returns() {
        // r = [ln(101/100), ln(102/101)] => доходности за периоды (0,1) и (1,2)
        let r = vec![(101.0f64 / 100.0).ln(), (102.0f64 / 101.0).ln()];
        // Агрегированная доходность за два периода: ln(102/100)
        let expected = (102.0f64 / 100.0).ln();
        assert_eq!(agg_log_return_from_log_returns(&r, 0, 2), Some(expected));
        // За один период с t=0: ln(101/100)
        assert_eq!(agg_log_return_from_log_returns(&r, 0, 1), Some(r[0]));
        // За один период с t=1: ln(102/101)
        assert_eq!(agg_log_return_from_log_returns(&r, 1, 1), Some(r[1]));
        // Горизонт 0
        assert_eq!(agg_log_return_from_log_returns(&r, 0, 0), Some(0.0));
        // Выход за пределы
        assert_eq!(agg_log_return_from_log_returns(&r, 0, 3), None);
        assert_eq!(agg_log_return_from_log_returns(&r, 2, 1), None);
    }

    #[test]
    fn test_empirical_kurtosis() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let k = empirical_kurtosis(&data).unwrap();
        // Для равномерного распределения эксцесс = -1.2
        assert!((k + 1.2).abs() < 1e-10);
    }

    #[test]
    fn test_hill_estimator() {
        // Искусственный пример
        let sorted = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let gamma = hill_estimator(&sorted).unwrap();
        assert!(gamma > 0.0);
    }

    // Тесты уровня 2
    #[test]
    fn test_cusum_mean_on_iid() {
        let data: Vec<f64> = (0..500).map(|_| rand::thread_rng().gen::<f64>()).collect();
        let (stat, p) = cusum_bwb_mean_test(&data, None, 200);
        println!("CUSUM mean: stat={:.3}, p={:.3}", stat, p);
        assert!(p > 0.01); // iid данные не должны отвергать H0
    }

    #[test]
    fn test_mf_dfa() {
        // Генерируем данные с долгой памятью (ARFIMA(0,0.4,0)) упрощённо
        let n = 2000;
        let mut rng = rand::thread_rng();
        let mut x = vec![0.0; n];
        for i in 1..n {
            x[i] = 0.8 * x[i - 1] + rng.gen::<f64>() * 0.1;
        }
        let res = mf_dfa(&x, -5.0, 5.0, 0.5, None, 1);
        println!("delta_alpha = {:.3}, delta_h = {:.3}", res.delta_alpha, res.delta_h);
        assert!(res.delta_alpha > 0.0);
    }

    #[test]
    fn test_variance_decay() {
        let data: Vec<f64> = (0..1000).map(|_| rand::thread_rng().gen::<f64>()).collect();
        let res = variance_decay_check(&data, 50, 500);
        for (m, ratio) in res.m_values.iter().zip(res.ratio.iter()) {
            println!("m={}, ratio={:.3}", m, ratio);
            // Для iid отношение должно быть около 1
            if *m > 5 {
                assert!((ratio - 1.0).abs() < 0.3);
            }
        }
    }

    // Тесты уровня 3
    #[test]
    fn test_compute_scale_estimate() {
        let y_r = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y_s = y_r.clone();
        let params = StsrvParams::new(2, 1, 3, 1.0, 0.1, 100);
        let k_scale = compute_scale_estimate(&y_r, &y_s, 2, &params);
        assert!((k_scale - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_preaverage_and_truncate() {
        let prices = vec![1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0];
        let params = StsrvParams::new(10, 5, 3, 2.0, 0.01, 100);
        let smoothed = preaverage_and_truncate(&prices, &params);
        assert_eq!(smoothed.len(), 5);
        assert!((smoothed[0] - 1.5).abs() < 1e-10);
        // Здесь порог большой, поэтому усечение не срабатывает
    }

    #[test]
    fn test_truncated_stsrv() {
        let y_r = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let y_s = y_r.clone();
        let params = StsrvParams::new(3, 1, 2, 10.0, 0.1, 100);
        let est = truncated_stsrv(&y_r, &y_s, &params);
        assert!((est - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_xi_calculation() {
        let params = StsrvParams::new(10, 2, 5, 1.0, 0.01, 1000);
        let expected = 1000.0 * (5.0_f64).powi(2) / ((8.0_f64).powi(3) * 0.01);
        assert!((params.xi() - expected).abs() < 1e-10);
    }
}

