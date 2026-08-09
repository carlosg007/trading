# trading

Backtesting environment.

## Layout

    mdlib/       lake access - returns bars, owns session/roll definitions
    data_pull/   vendor downloaders -> /mnt/backtest/raw -> /mnt/backtest/lake
    backtest/    engine: fills, costs, metrics
    strategies/  signal logic only
    scripts/     utilities

## Data

    /mnt/backtest/raw/{futures,equities,options,crypto,forex}   original vendor files
    /mnt/backtest/lake/...                                canonical Parquet
    /mnt/backtest/reference/                              specs, calendars
    /mnt/backtest/artifacts/                              backtest outputs

Futures are stored as a back-adjusted continuous series (symbol=ES),
rolled on volume crossover. Changing roll rules requires re-ingest.
