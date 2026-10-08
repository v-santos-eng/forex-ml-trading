# Forex ML Trading

An end-to-end research and execution stack for machine-learning-driven forex trading on MetaTrader 5: data preparation, feature engineering, model training and validation, risk management, order execution, and monitoring.

This is a personal research project. It is not financial advice, it makes no performance claims, and it should only ever be pointed at a demo account unless you have validated it yourself.

## The problem

Most retail trading bots fail for reasons that have little to do with the model: look-ahead bias in the backtest, position sizes that ignore volatility, orders rejected by the broker for invalid stops, and no record of why a trade was taken. This project treats those as the main engineering work and the predictor as one replaceable part.

## What it does

- **Data**: loads 1-minute historical CSVs or pulls bars from MT5 / Yahoo Finance, validates and resamples them (`csv_data_loader.py`, `data_validator.py`, `fx_preprocessor.py`).
- **Features and labels**: technical indicators, fractional differentiation, triple-barrier labelling and meta-labelling (`src/advanced_ml/`, `advanced_feature_engineering.py`).
- **Models**: Random Forest, Gradient Boosting and XGBoost classifiers/regressors per symbol and timeframe, with hyperparameter search (`fx_model_trainer.py`, `fx_hyperparameter_optimizer.py`, `ml_predictor_local.py`).
- **Validation**: purged k-fold cross-validation, walk-forward analysis and drift detection, to keep information from leaking across the train/test boundary (`purged_cross_validation.py`, `walk_forward_analysis.py`, `model_drift_detection.py`).
- **Risk**: ATR-based stops, Kelly-fraction position sizing, risk budgeting and exposure limits (`advanced_risk_system.py`, `kelly_position_sizing.py`, `risk_budgeting.py`).
- **Execution**: native MetaTrader 5 integration with reconnect handling and stop-level checks before an order is sent (`mt5_integration.py`, `auto_trader.py`, `auto_reconnect_manager.py`).
- **Observability**: every decision is logged to SQLite with the inputs that produced it; a Tkinter dashboard shows positions and system health (`detailed_decision_logger.py`, `mt5_dashboard.py`, `performance_monitor.py`).
- **LLM agents (optional)**: Agno agents backed by Claude or a local Ollama model that summarise model output and market context (`agno_*.py`, `forex_agent.py`).

## Stack

Python · pandas · NumPy · scikit-learn · XGBoost · LightGBM · MetaTrader5 · SQLite · Tkinter · Agno · Ollama / Anthropic API

## Layout

```
*.py                     core modules (flat, import each other directly)
src/advanced_ml/         financial ML methods: triple barrier, meta-labelling, purged CV, fracdiff
production_deployment/   packaged variant with launcher, scheduler, recovery and dashboards
examples/                runnable demos
tests/                   standalone verification scripts (not a pytest suite)
docs/                    setup, integration and operations guides (Portuguese)
```

## Running it

Requires Python 3.10+ and, for live data or execution, Windows with a MetaTrader 5 terminal installed.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt        # or requirements_free.txt for the no-API-key subset
copy .env.example .env                 # then fill in your values
```

All credentials are read from environment variables (`MT5_LOGIN`, `MT5_PASSWORD`, `MT5_SERVER`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`); nothing is stored in the repository. Load `.env` into your shell before running.

```bash
python examples/demo_without_api.py     # no keys, no broker
python train_ml_with_local_data.py      # train on local CSV data (set FX_DATA_DIR)
python mt5_connection_test.py           # check the MT5 connection
python mt5_main_launcher.py             # full system: MT5 + dashboard + ML + risk
```

Verification scripts run on their own, for example `python tests/test_risk_system_simple.py`. Those with `mt5` in the name need a running terminal.

## Data

Historical data, trained models, logs and databases are not in the repository; they are generated locally and ignored by git. The training scripts expect 1-minute CSVs under `FX_DATA_DIR` (default `data/FX-1-Minute-Data`).

## Status and limitations

- No live track record is claimed; treat every result as a backtest.
- The codebase grew by iteration and several modules overlap (`ml_predictor_*`, `fx_model_evaluator_*`); the flat layout is kept so existing imports work.
- `TA-Lib` needs its native library installed separately on Windows.
- Documentation under `docs/` is in Portuguese.

## References

The validation and labelling methods follow Marcos López de Prado, *Advances in Financial Machine Learning* (Wiley, 2018), and Stefan Jansen, *Machine Learning for Algorithmic Trading* (Packt, 2020).

## License

MIT. See [LICENSE](LICENSE).
