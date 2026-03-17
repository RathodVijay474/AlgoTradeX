# AlgoTradeX Experiment Report

## Dataset Summary
- Ticker: `AAPL`
- Date Range: `2015-02-02` to `2025-12-30`
- Total Rows: `2745`
- Train Rows: `1921`
- Validation Rows: `411`
- Test Rows: `413`
- Feature Count: `17`

## Configuration
- Algorithm: `ppo`
- Initial Capital: `100000.0`
- Transaction Cost: `0.001`
- Slippage: `0.0005`
- Reward Mode: `hybrid`

## Validation Metrics
| Metric | Value |
| --- | ---: |
| total_return | 0.209561 |
| annualized_return | 0.124050 |
| annualized_volatility | 0.251364 |
| sharpe_ratio | 0.589129 |
| sortino_ratio | 0.927442 |
| calmar_ratio | 0.642572 |
| max_drawdown | -0.193053 |
| win_rate | 0.501217 |
| average_exposure | 0.980052 |
| turnover | 9.338411 |
| trade_count | 407.000000 |
| benchmark_return | 0.180735 |
| excess_return | 0.028825 |

## Test Metrics
| Metric | Value |
| --- | ---: |
| total_return | 0.459559 |
| annualized_return | 0.260220 |
| annualized_volatility | 0.285541 |
| sharpe_ratio | 0.949676 |
| sortino_ratio | 1.264545 |
| calmar_ratio | 0.778646 |
| max_drawdown | -0.334195 |
| win_rate | 0.556901 |
| average_exposure | 0.975304 |
| turnover | 8.127258 |
| trade_count | 406.000000 |
| benchmark_return | 0.494363 |
| excess_return | -0.034804 |

## Notes
- Metrics are produced on chronological holdout splits to reduce look-ahead bias.
- Reward includes penalties for volatility, turnover, and drawdown.
- Transaction costs and slippage are applied inside the environment before next-bar PnL is realized.