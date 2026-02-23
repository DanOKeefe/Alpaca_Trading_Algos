import io


def text_report(result):
    """Generate a plain-text performance report."""
    m = result.metrics
    lines = [
        f"=== {result.strategy_name} ===",
        f"Period:        {result.config.start_date} to {result.config.end_date}",
        f"Initial Capital: ${result.config.initial_capital:,.0f}",
        "",
        f"Total Return:      {m.get('total_return', 0):>8.2%}",
        f"Annualized Return: {m.get('annualized_return', 0):>8.2%}",
        f"Sharpe Ratio:      {m.get('sharpe_ratio', 0):>8.2f}",
        f"Max Drawdown:      {m.get('max_drawdown', 0):>8.2%}",
        f"Ann. Volatility:   {m.get('annualized_volatility', 0):>8.2%}",
        "",
        f"Final Value:       ${_final_value(result):>12,.2f}",
    ]
    return "\n".join(lines)


def comparison_table(results):
    """Generate a text comparison table for multiple backtest results."""
    header = (
        f"{'Strategy':<25} {'Period':<25} "
        f"{'Total':>8} {'CAGR':>8} {'Sharpe':>7} "
        f"{'Max DD':>8} {'Vol':>8}"
    )
    sep = "-" * len(header)
    lines = [header, sep]

    for r in results:
        m = r.metrics
        period = f"{r.config.start_date} to {r.config.end_date}"
        lines.append(
            f"{r.strategy_name:<25} {period:<25} "
            f"{m.get('total_return', 0):>7.2%} "
            f"{m.get('annualized_return', 0):>7.2%} "
            f"{m.get('sharpe_ratio', 0):>7.2f} "
            f"{m.get('max_drawdown', 0):>7.2%} "
            f"{m.get('annualized_volatility', 0):>7.2%}"
        )

    return "\n".join(lines)


def html_report(results):
    """Generate an HTML performance report for one or more results."""
    buf = io.StringIO()
    buf.write("<!DOCTYPE html>\n<html><head>\n")
    buf.write("<title>Backtest Report</title>\n")
    buf.write("<style>\n")
    buf.write("body { font-family: sans-serif; margin: 2em; }\n")
    buf.write("table { border-collapse: collapse; margin: 1em 0; }\n")
    buf.write("th, td { border: 1px solid #ccc; padding: 8px 12px; "
              "text-align: right; }\n")
    buf.write("th { background: #f5f5f5; }\n")
    buf.write("td:first-child, th:first-child { text-align: left; }\n")
    buf.write("</style>\n</head><body>\n")
    buf.write("<h1>Backtest Performance Report</h1>\n")

    buf.write("<table>\n<tr>")
    for col in ["Strategy", "Period", "Total Return", "CAGR",
                "Sharpe", "Max Drawdown", "Volatility", "Final Value"]:
        buf.write(f"<th>{col}</th>")
    buf.write("</tr>\n")

    for r in results:
        m = r.metrics
        period = f"{r.config.start_date} &ndash; {r.config.end_date}"
        buf.write("<tr>")
        buf.write(f"<td>{r.strategy_name}</td>")
        buf.write(f"<td>{period}</td>")
        buf.write(f"<td>{m.get('total_return', 0):.2%}</td>")
        buf.write(f"<td>{m.get('annualized_return', 0):.2%}</td>")
        buf.write(f"<td>{m.get('sharpe_ratio', 0):.2f}</td>")
        buf.write(f"<td>{m.get('max_drawdown', 0):.2%}</td>")
        buf.write(f"<td>{m.get('annualized_volatility', 0):.2%}</td>")
        buf.write(f"<td>${_final_value(r):,.2f}</td>")
        buf.write("</tr>\n")

    buf.write("</table>\n</body></html>")
    return buf.getvalue()


def _final_value(result):
    if len(result.equity_curve) > 0:
        return result.equity_curve.iloc[-1]
    return result.config.initial_capital
