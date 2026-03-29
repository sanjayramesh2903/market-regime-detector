"""Color constants and helper functions for Market Regime Detector."""

REGIME_COLORS = {
    "bull": {"bg": "#0a1a0a", "border": "#00d26a", "text": "#00d26a", "plotly": "#00d26a"},
    "bear": {"bg": "#1a0a0a", "border": "#ff3b3b", "text": "#ff3b3b", "plotly": "#ff3b3b"},
    "volatile": {"bg": "#1a150a", "border": "#ff8c00", "text": "#ff8c00", "plotly": "#ff8c00"},
}

SUGGESTED_TICKERS = [
    "SPY", "QQQ", "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "JPM",
    "GLD", "TLT", "IWM", "DIA", "VTI", "EEM", "XLF", "XLE", "XLK", "XLV",
    "BTC-USD", "ETH-USD", "NFLX", "AMD", "INTC", "BA", "DIS", "V", "MA", "WMT",
    "PG", "JNJ", "UNH", "HD", "KO", "PEP", "COST", "ABBV", "MRK", "LLY",
    "CRM", "ORCL", "ADBE", "CSCO", "QCOM", "TXN", "AVGO", "AMAT", "MU", "SLV",
]


def suggest_tickers(query: str, limit: int = 5) -> list[str]:
    """Return top matching tickers from the suggested list."""
    q = query.upper().strip()
    if not q:
        return SUGGESTED_TICKERS[:limit]
    return [t for t in SUGGESTED_TICKERS if q in t][:limit]
