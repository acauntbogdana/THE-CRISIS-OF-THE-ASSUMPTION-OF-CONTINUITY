// Константы конфигурации и словарь единиц измерения.

use crate::equations::StsrvParams;

// Пути к данным
pub const DEEP_DIR: &str = r"Ваш путь\IEX\DEEP";
pub const FUND_DIR: &str = r"Ваш путь\IEX\Fundamental\parquet_by_year";
pub const TICKERS_FILE: &str = r"Ваш путь\IEX\Fundamental\expanded_tickers.parquet";
pub const SIC_FILE: &str = r"Ваш путь\IEX\Fundamental\sic_codes.parquet";
pub const MACRO_FILE: &str = r"Ваш путь\IEX\Macro\macro_indicators_2018_2022.parquet";
pub const OUTPUT_DIR: &str = r"Ваш путь";

// Параметры анализа
pub const MIN_VOLUME: u64 = 1_000_000;
pub const BAR_INTERVAL_MIN: i64 = 1;
pub const HORIZONS: &[usize] = &[1, 5, 15, 30, 60, 120, 240, 480, 1440];
pub const JUMP_THRESHOLD_MULT: f64 = 3.0;

pub const STSRV_PARAMS: StsrvParams = StsrvParams {
    k: 20,
    j: 5,
    m_preavg: 5,
    threshold: 0.001,
    max_tau_step: 1.0 / (252.0 * 390.0),
    n_blocks: 390,
};

pub const SELECTED_MACRO_COLS: &[&str] = &[
    "vix_fred_ffill",
    "ted_spread_ffill",
    "fed_funds_rate_ffill",
    "treasury_10y_ffill",
    "economic_policy_uncertainty_ffill",
    "CPI_headline_ffill",
    "PCE_headline_ffill",
    "PPI_all_commodities_ffill",
    "unemployment_rate_ffill",
    "industrial_production_ffill",
    "nonfarm_payrolls_ffill",
    "umich_consumer_sentiment_ffill",
    "retail_sales_ffill",
    "dollar_index_broad_ffill",
    "real_effective_exchange_rate_ffill",
    "oil_wti_ffill",
    "gold_ffill",
    "copper_ffill",
    "m2_money_supply_ffill",
    "gdp_real_ffill",
    "default_spread_baa_aaa_ffill",
    "yield_curve_10y_2y_ffill",
    "real_rate_10y_tips_ffill",
];

// Словарь единиц измерения
pub fn unit_for(col: &str) -> &'static str {
    let base = col.strip_suffix("_ffill").unwrap_or(col);
    match base {
        // IEX DEEP
        "msg_type" => "категория: тип сообщения IEX DEEP",
        "timestamp" => "наносекунды с полуночи (IEX epoch, UTC)",
        "symbol" => "тикер, строка до 8 симв.",
        "flags" => "битовые флаги сообщения (0-255)",
        "side" => "сторона: B=Buy / S=Sell",
        "price" => "USD за акцию",
        "size" => "акции, шт.",
        "trade_id" => "ID сделки (порядковый номер IEX)",
        "is_trade_break" => "булево: true = отмена сделки",
        "price_type" => "категория: opening/closing",
        "auction_type" => "категория: тип аукциона",
        "paired_shares" => "акции, шт.",
        "reference_price" => "USD за акцию",
        "indicative_clearing_price" => "USD за акцию",
        "imbalance_shares" => "акции, шт.",
        "imbalance_side" => "сторона дисбаланса B/S",
        "extension_number" => "порядковый номер продления аукциона",
        "scheduled_auction_time" => "секунды с полуночи (epoch seconds)",
        "auction_book_clearing_price" => "USD за акцию",
        "collar_reference_price" => "USD за акцию",
        "lower_auction_collar" | "upper_auction_collar" => "USD за акцию",
        "round_lot_size" => "акции, шт.",
        "adj_poc_price" => "USD за акцию",
        "luld_tier" => "категория: LULD Tier (0/1/2)",
        "trading_status" => "категория: статус торгов",
        "reason" => "код причины, 4 симв.",
        "halt_status" => "категория: halted/not_halted",
        "short_sale_status" => "код статуса Reg SHO",
        "short_sale_detail" => "категория: none/activated/continued",
        "retail_indicator" => "категория: none/buy/sell/buy_and_sell",
        "security_event" => "категория: событие по инструменту",
        "system_event" => "категория: системное событие фида",

        // SEC Fundamental
        "adsh" => "accession number отчёта (ID)",
        "tag" => "имя XBRL-тега",
        "version" => "таксономия/версия тега",
        "coreg" => "coregistrant (обычно пусто)",
        "ddate" => "дата данных (конец периода), timestamp",
        "qtrs" => "число кварталов: 0=instant,1=quarter,4=год",
        "uom" => "единица измерения поля value (USD, shares, pure, USD/shares...)",
        "value" => "числовое значение факта, В ЕДИНИЦАХ КОЛОНКИ uom",
        "footnote" => "текст сноски к факту",
        "cik" => "SEC Central Index Key (ID компании)",
        "name" => "название компании",
        "sic" => "SIC-код отрасли (строка!)",
        "countryba" => "страна штаб-квартиры (код)",
        "stprba" => "штат/провинция штаб-квартиры",
        "cityba" => "город штаб-квартиры",
        "fye" => "конец фин. года, MMDD (строка)",
        "form" => "тип формы SEC (10-K, 10-Q, ...)",
        "period" => "отчётный период, YYYYMMDD (int)",
        "fy" => "финансовый год",
        "fp" => "квартал: Q1..Q4/FY",
        "filed" => "дата подачи, YYYYMMDD (int)",
        "afs" => "статус ускоренной подачи (accelerated filer status)",
        "wksi" => "well-known seasoned issuer: 0/1",
        "stmt" => "часть отчёта: BS/IS/CF/EQ/CI/UN",
        "plabel" => "подпись строки в отчёте (presentation label)",
        "line" => "номер строки в отчёте (ВНИМАНИЕ: тип различается между годами)",
        "negating" => "флаг инверсии знака при отображении",
        "tlabel" => "человекочитаемое название тега",
        "doc" => "текст-документация тега (XBRL definition)",
        "datatype" => "тип данных XBRL: monetary/shares/pure/...",
        "iord" => "instant или duration",
        "crdr" => "credit или debit",
        "quarter" => "метка квартала-источника, напр. '2018q1'",

        // Справочники
        "ticker" => "биржевой тикер",
        "source" => "источник пары CIK->ticker",
        "SIC Code" => "SIC-код (числовой)",
        "Office" => "офис SEC, курирующий отрасль",
        "Industry Title" => "название отрасли",

        // Макро
        "date" => "календарная дата (день)",
        "CPI_headline" | "CPI_core" | "PCE_headline" | "PCE_core"
            | "PPI_all_commodities" => "индекс цен, пункты (1982-84=100 и т.п.)",
        "breakeven_5y" | "breakeven_10y" | "forward_5y5y_inflation" => "% годовых",
        "fed_funds_rate" | "fed_funds_rate_daily" | "treasury_10y" | "treasury_2y"
            | "treasury_3m" | "real_rate_10y_tips" | "moodys_baa_yield"
            | "moodys_aaa_yield" => "% годовых",
        "yield_curve_10y_2y" | "yield_curve_10y_3m" | "baa_10y_spread"
            | "aaa_10y_spread" | "ted_spread" | "default_spread_baa_aaa"
            | "real_rate_10y_calc" => "п.п. (спред доходностей, % годовых)",
        "m2_money_supply" => "млрд USD",
        "gdp_nominal" | "gdp_real" => "млрд USD (GDPC1 — в ценах 2017 г.)",
        "industrial_production" => "индекс, пункты (2017=100)",
        "nonfarm_payrolls" => "тыс. занятых",
        "unemployment_rate" => "%",
        "initial_jobless_claims" => "число заявок, шт./неделя",
        "umich_consumer_sentiment" => "индекс, пункты",
        "retail_sales" => "млн USD",
        "nfci_chicago_fed" => "индекс (std. dev. от среднего)",
        "vix_fred" | "vix_yf" => "пункты волатильности (VIX)",
        "dollar_index_broad" | "dxy_yf" => "индекс, пункты",
        "real_effective_exchange_rate" => "индекс, пункты (2010=100)",
        "economic_policy_uncertainty" => "индекс EPU, пункты",
        "oil_wti" | "oil_brent" => "USD за баррель",
        "copper" => "USD за фунт (COMEX)",
        "gold" => "USD за тройскую унцию",
        "m2_yoy_growth_pct" | "gdp_real_yoy_growth_pct" => "% год к году",

        _ => "—",
    }
}
