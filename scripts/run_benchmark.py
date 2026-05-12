from pathlib import Path
import sys

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.features import build_training_data
from src.data.loaders import list_cycle_files, load_cycle_df, load_parameter_evolution, parse_cycle_number
from src.data.splits import split_cycles
from src.evaluation.benchmark import evaluate_cycle
from src.evaluation.metrics import summarize_metrics
from src.ml.residual_rf import train_residual_rf


def run_benchmark(max_files=60, max_rows=20000, val_ratio=0.2):
    files = list_cycle_files()[:max_files]
    train_files, val_files = split_cycles(files, parse_cycle_number, val_ratio=val_ratio)

    df_params = load_parameter_evolution()
    train_df = build_training_data(train_files, df_params, load_cycle_df, max_rows=max_rows)
    if train_df.empty:
        raise RuntimeError("Training data is empty. Check processed cycle files.")

    model = train_residual_rf(train_df)

    rows = []
    eval_files = val_files if val_files else files
    for path in eval_files:
        cycle = parse_cycle_number(path)
        if cycle is None:
            continue
        out = evaluate_cycle(model, path, cycle, df_params)
        rows.append(
            {
                "cycle": out["cycle"],
                "rmse_physics": out["physics"]["rmse"],
                "rmse_hybrid": out["hybrid"]["rmse"],
                "mae_physics": out["physics"]["mae"],
                "mae_hybrid": out["hybrid"]["mae"],
            }
        )

    df = summarize_metrics(rows)
    return df


def export_artifacts(df, out_dir="results"):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    detail_csv = out_path / "benchmark_cycle_metrics.csv"
    summary_csv = out_path / "benchmark_summary.csv"
    plot_box = out_path / "benchmark_rmse_boxplot.png"
    plot_line = out_path / "benchmark_rmse_by_cycle.png"

    df.to_csv(detail_csv, index=False)
    df.describe(include="all").to_csv(summary_csv)

    fig1, ax1 = plt.subplots(figsize=(7, 4))
    ax1.boxplot([df["rmse_physics"], df["rmse_hybrid"]], tick_labels=["Physics", "Hybrid"])
    ax1.set_title("RMSE Distribution by Model")
    ax1.set_ylabel("RMSE (degC)")
    fig1.tight_layout()
    fig1.savefig(plot_box, dpi=180)
    plt.close(fig1)

    df_sorted = df.sort_values("cycle")
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.plot(df_sorted["cycle"], df_sorted["rmse_physics"], marker="o", label="Physics RMSE")
    ax2.plot(df_sorted["cycle"], df_sorted["rmse_hybrid"], marker="o", label="Hybrid RMSE")
    ax2.set_xlabel("Cycle")
    ax2.set_ylabel("RMSE (degC)")
    ax2.set_title("RMSE by Validation Cycle")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(plot_line, dpi=180)
    plt.close(fig2)

    return {
        "detail_csv": str(detail_csv),
        "summary_csv": str(summary_csv),
        "plot_box": str(plot_box),
        "plot_line": str(plot_line),
    }


if __name__ == "__main__":
    df = run_benchmark()
    artifacts = export_artifacts(df)
    print(df.describe(include="all"))
    print("\nSaved artifacts:")
    for k, v in artifacts.items():
        print(f"- {k}: {v}")
