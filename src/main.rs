// Точка входа в приложение.

use std::time::Instant;
use anyhow::Result;
use env_logger::Env;
use dataset_report::analysis_core; // <-- импорт модуля из библиотеки

fn main() -> Result<()> {
    // Инициализация логгера (по умолчанию уровень WARN)
    env_logger::Builder::from_env(Env::default().default_filter_or("warn")).init();

    let start = Instant::now();
    analysis_core::run()?;
    println!("Общее время: {:.1} с", start.elapsed().as_secs_f32());
    Ok(())
}