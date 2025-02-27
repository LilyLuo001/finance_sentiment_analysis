Sentiment-Driven Market Microstructure Analysis
Overview
This project examines the impact of financial sentiment on market microstructure using FinBERT-based sentiment analysis and high-frequency market data. The goal is to analyze whether sentiment shocks affect liquidity, bid-ask spreads, order book depth, and execution costs in the short term. 


sentiment-market-microstructure/

│── data_pipeline.py        # Market & sentiment data collection

│── feature_engineering.py  # Liquidity, volatility, order book features

│── model.py                # FinBERT sentiment classification

│── execution.py            # Trade execution and market impact modeling

│── causality.py            # Granger causality & event study framework

│── experiment_tracking.py  # Logging & backtesting results

│── run_pipeline.py         # Full workflow execution

│── README.md               # Project documentation
