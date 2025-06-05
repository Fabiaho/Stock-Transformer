stock-transformer-predictor/
├── README.md
├── requirements.txt
├── setup.py
├── .gitignore
├── .env.example
├── pyproject.toml
│
├── config/
│   ├── __init__.py
│   ├── model_config.yaml
│   ├── data_config.yaml
│   └── training_config.yaml
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── collectors/
│   │   │   ├── __init__.py
│   │   │   ├── price_collector.py
│   │   │   ├── news_collector.py
│   │   │   └── economic_collector.py
│   │   ├── processors/
│   │   │   ├── __init__.py
│   │   │   ├── technical_indicators.py
│   │   │   ├── feature_engineering.py
│   │   │   └── preprocessor.py
│   │   ├── dataset.py
│   │   └── datamodule.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── transformer.py
│   │   ├── attention.py
│   │   ├── embeddings.py
│   │   └── lightning_module.py
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── callbacks.py
│   │   ├── losses.py
│   │   └── metrics.py
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── backtester.py
│   │   ├── financial_metrics.py
│   │   └── visualization.py
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logging.py
│       ├── io.py
│       └── helpers.py
│
├── scripts/
│   ├── download_data.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── hyperparameter_search.py
│   └── run_backtest.py
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_prototyping.ipynb
│   ├── 04_results_analysis.ipynb
│   └── 05_attention_visualization.ipynb
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── external/
│   └── .gitkeep
│
├── models/
│   ├── checkpoints/
│   ├── experiments/
│   └── .gitkeep
│
├── results/
│   ├── figures/
│   ├── reports/
│   ├── logs/
│   └── .gitkeep
│
├── tests/
│   ├── __init__.py
│   ├── test_data/
│   │   ├── test_collectors.py
│   │   ├── test_processors.py
│   │   └── test_dataset.py
│   ├── test_models/
│   │   ├── test_transformer.py
│   │   └── test_lightning_module.py
│   └── test_evaluation/
│       ├── test_backtester.py
│       └── test_metrics.py
│
├── docs/
│   ├── architecture.md
│   ├── data_sources.md
│   ├── model_details.md
│   └── evaluation_methodology.md
│
└── docker/
    ├── Dockerfile
    ├── docker-compose.yml
    └── requirements-docker.txt