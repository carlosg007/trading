"""
Portfolio-to-account mapping: which strategies trade which basket, on which
account, at what size.

This package sits ABOVE `backtest/` in the one-way dependency chain
(`agents` -> `strategies` -> `backtest` -> `mdlib` -> lake) and may read from
it — `config_loader` reconciles its asset metadata against
`backtest/specs.py`. Nothing in `backtest/` may import from here: a research
run must not depend on which live account a strategy would eventually be
routed to.
"""
