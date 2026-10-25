
"""Gemini API実験の統計分析と可視化。"""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results.csv"
FIGURES = ROOT / "figures"

FIGURES.mkdir(exist_ok=True)


def main():
    if not RESULTS.exists():
        print("results.csv が見つかりません")
        return

    with RESULTS.open(
        encoding="utf-8", newline=""
    ) as f:
        rows = [
            r for r in csv.DictReader(f)
            if r["status"] == "success"
            and r["output_tokens"] not in ("", "None")
            and r["elapsed"] not in ("", "None")
        ]

    if not rows:
        print("分析できるデータがありません")
        return

    # 条件別のデータをまとめる
    groups = defaultdict(list)
    plots = defaultdict(list)

    for r in rows:
        model = r["model"]
        question = r["question"]
        target = int(r["target"])
        output = float(r["output_tokens"])
        elapsed = float(r["elapsed"])

        groups[(model, question, target)].append(
            (output, elapsed)
        )
        plots[(model, question)].append(
            (output, elapsed)
        )

    # 1. トークン数・応答時間の平均と標準偏差
    summary = ROOT / "summary.csv"

    with summary.open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.writer(f)
        writer.writerow([
            "model", "question", "target",
            "n", "mean_output_tokens",
            "std_output_tokens",
            "mean_elapsed_seconds",
            "std_elapsed_seconds",
        ])

        for (model, question, target), data in sorted(
            groups.items()
        ):
            tokens = np.array([p[0] for p in data])
            times = np.array([p[1] for p in data])

            mean_tokens = np.mean(tokens)
            mean_time = np.mean(times)

            std_tokens = (
                np.std(tokens, ddof=1)
                if len(tokens) > 1 else np.nan
            )
            std_time = (
                np.std(times, ddof=1)
                if len(times) > 1 else np.nan
            )

            writer.writerow([
                model, question, target, len(data),
                round(float(mean_tokens), 3),
                round(float(std_tokens), 3),
                round(float(mean_time), 4),
                round(float(std_time), 4),
            ])

            print(
                f"{model} | {question} | "
                f"目標{target} | n={len(data)} | "
                f"実測平均={mean_tokens:.1f} tokens | "
                f"SD={std_tokens:.1f}"
            )

    # 2. モデルごとに散布図・回帰直線を作成
    models = sorted({key[0] for key in plots})

    for model in models:
        plt.figure(figsize=(9, 6))

        for question in ("knowledge", "reasoning"):
            points = plots.get((model, question), [])

            if not points:
                continue

            x = np.array([p[0] for p in points])
            y = np.array([p[1] for p in points])

            scatter = plt.scatter(
                x, y, label=f"{question} data",
                alpha=0.8
            )

            # 2点以上かつ異なるトークン数が必要
            if len(x) >= 2 and len(np.unique(x)) >= 2:
                slope, intercept = np.polyfit(x, y, 1)
                predicted = slope * x + intercept

                ss_res = np.sum((y - predicted) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)

                r2 = (
                    1 - ss_res / ss_tot
                    if ss_tot > 0 else np.nan
                )

                line_x = np.linspace(
                    x.min(), x.max(), 100
                )

                plt.plot(
                    line_x,
                    slope * line_x + intercept,
                    color=scatter.get_facecolor()[0],
                    label=f"{question} regression"
                )

                print(
                    f"{model} / {question}: "
                    f"t = {intercept:.4f} + "
                    f"{slope:.6f} × tokens, "
                    f"R² = {r2:.4f}"
                )

        plt.xlabel("Observed output tokens")
        plt.ylabel("Response time (seconds)")
        plt.title(f"Output tokens vs response time: {model}")
        plt.legend()
        plt.grid(alpha=0.25)
        plt.tight_layout()

        path = FIGURES / f"{model}_regression.png"
        plt.savefig(path, dpi=200)
        plt.close()

        print(f"グラフ保存: {path}")

    print(f"\n集計表保存: {summary}")


if __name__ == "__main__":
    main()
