Open Tasks & Immediate Milestones
- Clean Environment: Permanently verify all courses/ remnants are deleted and uv manages ~/src/trading/.venv.
- Refactor agents/tier1_master.py: Keep static AST security and single-shot synthesis with Gemini 2.5 Pro; remove obsolete multi-tier LLM supervisor calls.
- Standardize Pure Alpha Scoring: Ensure the Streamlit dashboard (dashboard/app.py) and engine output pure institutional metrics: Annualized Sharpe ($\ge 1.2$), Sortino, Calmar, Profit Factor ($\ge 1.5$), Win Rate, and Max Drawdown.
- Generate Baseline Strategy: Stage strategies/experimental/sma_crossover.py to confirm the array pipeline runs end-to-end against the Databento lake within the $\le 3.0\text{ GiB}$ RAM ceiling.

