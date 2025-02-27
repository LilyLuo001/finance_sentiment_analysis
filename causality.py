from statsmodels.tsa.stattools import grangercausalitytests

class CausalityAnalyzer:
    """Enhanced causality analysis with multiple methods"""
    def __init__(self, max_lag: int = 12):
        self.max_lag = max_lag
        
    def granger_test(self, x: pd.Series, y: pd.Series) -> dict:
        """Perform Granger causality test with multiple lags"""
        data = pd.concat([x, y], axis=1)
        results = grangercausalitytests(data, maxlag=self.max_lag, verbose=False)
        return {
            lag: {
                'ssr_ftest': test[0]['ssr_ftest'],
                'params_ftest': test[0]['params_ftest']
            } for lag, test in results.items()
        }

    def event_study(self, events: pd.DatetimeIndex, returns: pd.Series, window: int = 5) -> dict:
        """Perform event study analysis around specified dates"""
        cumulative_returns = []
        for event in events:
            start = event - pd.Timedelta(days=window)
            end = event + pd.Timedelta(days=window)
            window_returns = returns.loc[start:end]
            cumulative_returns.append(window_returns.cumsum())
            
        return {
            'mean_car': pd.concat(cumulative_returns).groupby(level=0).mean(),
            'std_car': pd.concat(cumulative_returns).groupby(level=0).std()
        }
