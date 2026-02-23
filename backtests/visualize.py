import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd


def plot_equity_curves(results, save_path=None):
    """Plot equity curves for one or more backtest results."""
    fig, ax = plt.subplots(figsize=(12, 6))

    for r in results:
        if len(r.equity_curve) > 0:
            ax.plot(r.equity_curve.index, r.equity_curve.values,
                    label=r.strategy_name)

    ax.set_title("Equity Curves")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return fig


def plot_drawdowns(results, save_path=None):
    """Plot drawdown chart for one or more backtest results."""
    fig, ax = plt.subplots(figsize=(12, 5))

    for r in results:
        if len(r.equity_curve) > 0:
            peak = r.equity_curve.cummax()
            drawdown = (r.equity_curve - peak) / peak
            ax.fill_between(drawdown.index, drawdown.values, 0,
                            alpha=0.3, label=r.strategy_name)
            ax.plot(drawdown.index, drawdown.values, linewidth=0.8)

    ax.set_title("Drawdowns")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return fig


def plot_weight_allocation(result, save_path=None, top_n=10):
    """Plot stacked area chart of weight allocation over time."""
    if not result.weights_history:
        return None

    weights_df = pd.DataFrame(result.weights_history).T.fillna(0)

    # Show only the top_n assets by average weight
    avg_weights = weights_df.mean().sort_values(ascending=False)
    top_assets = avg_weights.head(top_n).index.tolist()
    other = weights_df.drop(columns=top_assets, errors="ignore").sum(axis=1)
    plot_df = weights_df[top_assets].copy()
    if other.sum() > 0:
        plot_df["Other"] = other

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.stackplot(plot_df.index, plot_df.T.values,
                 labels=plot_df.columns.tolist(), alpha=0.8)
    ax.set_title(f"Weight Allocation — {result.strategy_name}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Weight")
    ax.set_ylim(0, 1)
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=8)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig
