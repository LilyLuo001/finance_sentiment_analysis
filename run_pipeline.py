import argparse
import yaml
from pathlib import Path

def configure_parser() -> argparse.ArgumentParser:
    """Enhanced command line interface with config file support"""
    parser = argparse.ArgumentParser(description="Sentiment-Market Trading Pipeline")
    parser.add_argument("--ticker", type=str, required=True, help="Stock ticker symbol")
    parser.add_argument("--config", type=Path, default="config.yaml", help="Path to config file")
    parser.add_argument("--start", type=str, default="2020-01-01", help="Start date")
    parser.add_argument("--end", type=str, default="2023-01-01", help="End date")
    parser.add_argument("--backtest", action='store_true', help="Enable backtesting mode")
    parser.add_argument("--live", action='store_true', help="Enable live trading mode")
    return parser

def load_config(config_path: Path) -> dict:
    """Load configuration parameters from YAML file"""
    with open(config_path) as f:
        return yaml.safe_load(f)

def main():
    parser = configure_parser()
    args = parser.parse_args()
    config = load_config(args.config)
    
    # Initialize pipeline components
    data_pipeline = DataPipeline(args.ticker)
    feature_engineer = FeatureEngineeringPipeline(args.ticker)
    tracker = ExperimentTracker(f"{args.ticker}_experiments")
    
    # Execute data pipeline
    market_data = data_pipeline.fetch_market_data(args.start, args.end)
    sentiment_data = data_pipeline.load_sentiment_data(args.start, args.end)
    
    # Feature engineering
    processed_data = feature_engineer.pipeline.fit_transform(
        market_data.join(sentiment_data)
    )
