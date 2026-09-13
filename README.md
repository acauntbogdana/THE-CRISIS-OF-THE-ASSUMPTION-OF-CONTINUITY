# dataset-report

Анализ внутридневных рыночных данных на Rust: бары из IEX DEEP, уровни I–V,
логистическая регрессия вероятности скачков, графики.

## Требования

- Rust 1.75+ (edition 2021)
- Parquet-данные: IEX DEEP, SEC Fundamental, макро, справочник тикеров

## Сборка и запуск
///
cargo build --release
cargo run --release
