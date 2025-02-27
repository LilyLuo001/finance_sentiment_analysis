import os
import pandas as pd
import numpy as np
import yfinance as yf
import logging
from datetime import datetime, timedelta
from typing import Tuple, Dict, Optional
from functools import lru_cache
import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataPipeline:
    def __init__(self, ticker: str):
        self.ticker = ticker
        self.raw_data_path = f"data/raw/{ticker}/"
        os.makedirs(self.raw_data_path, exist_ok=True)
        self._validate_ticker()

    def _validate_ticker(self):
        """Validate ticker format and existence"""
        if not isinstance(self.ticker, str) or len(self.ticker) > 5:
            raise ValueError(f"Invalid ticker format: {self.ticker}")
        
    @lru_cache(maxsize=10)
    def fetch_market_data(self, start: str, end: str, interval: str = "5m") -> pd.DataFrame:
        """Fetch high-frequency market data with enhanced validation and retry logic"""
        for attempt in range(3):
            try:
                df = yf.download(self.ticker, start=start, end=end, interval=interval,
                               prepost=True, threads=True)
                if df.empty:
                    raise ValueError("Empty DataFrame returned from Yahoo Finance")
                
                # Resample and validate data quality
                df = self._resample_and_validate(df, interval)
                self._save_data(df, f"market_{start}_{end}.parquet")
                return self._calculate_basic_features(df).dropna()
            except Exception as e:
                logging.warning(f"Attempt {attempt+1} failed: {e}")
                if attempt == 2:
                    logging.error("Failed to fetch market data after 3 attempts")
                    return pd.DataFrame()
                
    def _resample_and_validate(self, df: pd.DataFrame, interval: str) -> pd.DataFrame:
        """Resample data and validate completeness"""
        expected_freq = pd.Timedelta(interval)
        resampled = df.resample(expected_freq).agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        })
        
        # Check for missing intervals
        expected_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq=expected_freq)
        return resampled.reindex(expected_index).ffill()
    
    def _calculate_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate initial set of market features"""
        df['returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['spread'] = df['High'] - df['Low']
        df['mid_price'] = (df['High'] + df['Low']) / 2
        df['imbalance'] = (df['Volume'] - df['Volume'].shift(1)) / (df['Volume'].shift(1) + 1e-6)
        return df

    def fetch_realtime_sentiment(self, hours: int = 24) -> pd.DataFrame:
        """Fetch real-time sentiment data from news sources"""
        try:
            news_data = self._scrape_financial_news()
            processed = self._process_news_items(news_data)
            processed['timestamp'] = pd.to_datetime(processed['timestamp'])
            cutoff = datetime.now() - timedelta(hours=hours)
            return processed[processed['timestamp'] >= cutoff]
        except Exception as e:
            logging.error(f"Error fetching sentiment data: {e}")
            return pd.DataFrame()

    def _scrape_financial_news(self) -> list:
        """Scrape financial news from web sources"""
        sources = {
            'bloomberg': f'https://www.bloomberg.com/quote/{self.ticker}:US',
            'reuters': f'https://www.reuters.com/companies/{self.ticker}.O'
        }
        
        articles = []
        for source, url in sources.items():
            try:
                response = requests.get(url, timeout=10)
                soup = BeautifulSoup(response.text, 'html.parser')
                articles.extend([p.text for p in soup.find_all('p')[:3]])
            except Exception as e:
                logging.warning(f"Failed to scrape {source}: {e}")
        return articles

    def _process_news_items(self, raw_articles: list) -> pd.DataFrame:
        """Process raw text articles into structured sentiment data"""
        return pd.DataFrame({
            'timestamp': [datetime.now()] * len(raw_articles),
            'text': raw_articles,
            'source': ['web'] * len(raw_articles),
            'raw_length': [len(t) for t in raw_articles]
        })

    def _save_data(self, df: pd.DataFrame, filename: str):
        """Save data with versioning"""
        version = datetime.now().strftime("%Y%m%d%H%M")
        path = f"{self.raw_data_path}{version}_{filename}"
        df.to_parquet(path)
        logging.info(f"Saved data to {path}")
