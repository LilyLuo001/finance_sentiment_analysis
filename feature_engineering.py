from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
import numba
import talib

class FeatureEngineeringPipeline:
    def __init__(self, ticker: str):
        self.ticker = ticker
        self.pipeline = self._build_pipeline()
        
    def _build_pipeline(self) -> Pipeline:
        """Create complete feature engineering pipeline"""
        return Pipeline([
            ('technical', TechnicalFeatureGenerator()),
            ('sentiment', SentimentFeatureGenerator()),
            ('post_process', PostProcessingTransformer())
        ])

class TechnicalFeatureGenerator(BaseEstimator, TransformerMixin):
    def __init__(self, lookback_windows: tuple = (5, 15, 60)):
        self.lookback_windows = lookback_windows

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate comprehensive technical features"""
        X = self._price_features(X)
        X = self._volume_features(X)
        X = self._volatility_features(X)
        X = self._momentum_features(X)
        X = self._liquidity_features(X)
        return X.dropna()

    def _price_features(self, df):
        """Calculate price-based technical indicators"""
        df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
        df['ADX'] = talib.ADX(df['High'], df['Low'], df['Close'], timeperiod=14)
        df['OBV'] = talib.OBV(df['Close'], df['Volume'])
        return df

    def _volume_features(self, df):
        """Calculate volume-based features"""
        for w in self.lookback_windows:
            df[f'vol_ma_{w}'] = df['Volume'].rolling(w).mean()
            df[f'vol_ratio_{w}'] = df['Volume'] / df[f'vol_ma_{w}']
        return df

    def _volatility_features(self, df):
        """Calculate volatility metrics"""
        returns = df['returns'].dropna()
        df['realized_vol'] = returns.rolling(20).std() * np.sqrt(252)
        df['parkinson_vol'] = (np.log(df['High']/df['Low'])**2).rolling(20).mean() * np.sqrt(252/(4*np.log(2)))
        return df

class SentimentFeatureGenerator(BaseEstimator, TransformerMixin):
    def __init__(self, lookback_windows: tuple = (1, 4, 12)):
        self.lookback_windows = lookback_windows  # In hours

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate sentiment features from raw text data"""
        if 'sentiment_score' not in X.columns:
            X['sentiment_score'] = 0.5  # Placeholder for actual sentiment analysis
        
        # Sentiment momentum features
        for w in self.lookback_windows:
            X[f'sentiment_ma_{w}h'] = X['sentiment_score'].rolling(w).mean()
            X[f'sentiment_roc_{w}h'] = X['sentiment_score'].pct_change(w)
            
        # Sentiment volatility
        X['sentiment_vol'] = X['sentiment_score'].rolling(24).std()
        return X

class PostProcessingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')

    def fit(self, X, y=None):
        self.scaler.fit(X.select_dtypes(include=np.number))
        return self

    def transform(self, X):
        X = X.copy()
        numeric_cols = X.select_dtypes(include=np.number).columns
        X[numeric_cols] = self.imputer.transform(X[numeric_cols])
        X[numeric_cols] = self.scaler.transform(X[numeric_cols])
        return X
