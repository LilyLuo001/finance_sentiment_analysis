import numpy as np
from scipy.stats import skew, kurtosis

class Backtester:
    """Enhanced backtester with transaction costs and slippage modeling"""
    def __init__(self, initial_capital=1e6, commission=0.0001, slippage=0.0005):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions = []
        self.portfolio_value = []
        self.commission = commission
        self.slippage = slippage
        self.trade_log = []

    def execute_trade(self, price: float, signal: float, position_size: float):
        """Execute trade with cost modeling"""
        if signal not in [-1, 0, 1]:
            raise ValueError("Invalid trading signal")
            
        # Calculate slippage impact
        executed_price = price * (1 + np.random.normal(0, self.slippage))
        trade_value = signal * position_size * executed_price
        
        # Calculate commissions
        commission_cost = abs(trade_value) * self.commission
        net_trade_value = trade_value + commission_cost
        
        # Update portfolio state
        self.capital -= net_trade_value
        self.positions.append(signal * position_size)
        current_value = self.capital + sum(self.positions) * executed_price
        self.portfolio_value.append(current_value)
        
        # Log trade details
        self.trade_log.append({
            'timestamp': datetime.now(),
            'price': executed_price,
            'signal': signal,
            'size': position_size,
            'commission': commission_cost,
            'slippage': executed_price - price
        })

    def calculate_metrics(self) -> dict:
        """Compute comprehensive performance metrics"""
        returns = np.diff(self.portfolio_value) / self.portfolio_value[:-1]
        
        metrics = {
            'sharpe_ratio': self._annualized_sharpe(returns),
            'sortino_ratio': self._sortino_ratio(returns),
            'max_drawdown': self._max_drawdown(),
            'profit_factor': self._profit_factor(),
            'skewness': skew(returns),
            'kurtosis': kurtosis(returns),
            'win_rate': self._win_rate(),
            'avg_trade_return': np.mean(returns),
            'total_commission': sum(t['commission'] for t in self.trade_log)
        }
        return metrics

    def _annualized_sharpe(self, returns: np.ndarray) -> float:
        if len(returns) < 2 or np.std(returns) == 0:
            return 0
        return np.sqrt(252*24*12) * np.mean(returns) / np.std(returns)

    def _sortino_ratio(self, returns: np.ndarray) -> float:
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0 or np.std(downside_returns) == 0:
            return 0
        return np.sqrt(252*24*12) * np.mean(returns) / np.std(downside_returns)

    def _max_drawdown(self) -> float:
        peak = self.initial_capital
        max_dd = 0
        for value in self.portfolio_value:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
        return max_dd

    def _profit_factor(self) -> float:
        gains = sum(t['size']*t['price'] for t in self.trade_log if t['signal'] > 0)
        losses = sum(t['size']*t['price'] for t in self.trade_log if t['signal'] < 0)
        return gains / losses if losses != 0 else float('inf')

    def _win_rate(self) -> float:
        winning_trades = sum(1 for t in self.trade_log if t['signal']*(t['price'] - t['price']) > 0)
        return winning_trades / len(self.trade_log) if self.trade_log else 0
