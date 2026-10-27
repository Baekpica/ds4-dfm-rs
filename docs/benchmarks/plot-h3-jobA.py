#!/usr/bin/env python3
"""Plot job A both ways; requires matplotlib, as the rest of this directory does.

Data: docs/benchmarks/h3-jobA-2026-10-27/h3-jobA-both-arms.csv, which holds both
arms measured on one binary with H3_DIT_ATTENTION_MMA as the only difference.
Same styling as plot-h3-attention.py: grey for the scalar kernel, teal for the
tensor core, values on the bars, and a footnote stating what the rows are.
"""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DATA = Path(__file__).with_name("h3-jobA-2026-10-27")
COLORS = {"scalar": "#687789", "tensor": "#087f8c"}
LABELS = {"scalar": "Scalar kernel", "tensor": "Tensor core"}
HEADLINE = (("DENOISE_2evals", "Denoise (2 evals)"), ("wall", "Wall, whole job"))
PHASES = (("TEXT", "Text"), ("LOAD", "Load"), ("DENOISE_2evals", "Denoise"),
          ("AUDIO", "Audio"), ("FRAMES", "Frames (VAE)"), ("VIDEO", "Video"))


def load():
    rows = {}
    with (DATA / "h3-jobA-both-arms.csv").open() as source:
        for row in csv.DictReader(source):
            if row["scope"] != "jobA" or row["arm"] not in LABELS:
                continue
            try:
                rows[(row["arm"], row["metric"])] = float(row["value"])
            except ValueError:
                continue    # the render hashes share this file and are not numbers
    return rows


def bars(ax, entries, rows, log=False, unit="s"):
    width = 0.38
    for index, (metric, heading) in enumerate(entries):
        for offset, arm in ((-0.5, "scalar"), (0.5, "tensor")):
            value = rows[(arm, metric)]
            left = index + offset * width
            ax.bar(left, value, width, color=COLORS[arm], alpha=0.9,
                   label=LABELS[arm] if index == 0 else None)
            text = f"{value:,.1f} {unit}" if value >= 1 else f"{value:.3f} {unit}"
            ax.annotate(text, (left, value), textcoords="offset points",
                        xytext=(0, 3), ha="center", fontsize=8.5)
    if log:
        ax.set_yscale("log")
    ax.set_xticks(range(len(entries)))
    ax.set_xticklabels([heading for _, heading in entries])
    ax.grid(axis="y", alpha=0.2, which="both")
    ax.set_axisbelow(True)


def plot():
    rows = load()
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.facecolor": "white"})
    fig, axes = plt.subplots(2, 1, figsize=(11, 8.5))
    fig.suptitle("MiniMax-H3 job A · scalar vs tensor core", fontsize=18,
                 fontweight="bold", y=0.985)
    fig.text(0.5, 0.945,
             "One DGX Spark / GB10 · 768x768 · 107 frames · 2 evaluations · seed 42 · 2026-10-27",
             ha="center", color="#4b5563")

    denoise = rows[("scalar", "DENOISE_2evals")] / rows[("tensor", "DENOISE_2evals")]
    wall = rows[("scalar", "wall")] / rows[("tensor", "wall")]
    bars(axes[0], HEADLINE, rows)
    axes[0].set_title(f"Headline · denoise {denoise:.2f}x shorter, wall {wall:.2f}x",
                      loc="left", fontsize=13, pad=10)
    axes[0].set_ylabel("Seconds")
    axes[0].legend(loc="upper right", framealpha=0.92)
    axes[0].margins(y=0.18)

    bars(axes[1], PHASES, rows, log=True)
    drift = abs(rows[("scalar", "FRAMES")] - rows[("tensor", "FRAMES")])
    axes[1].set_title("Phase walls · only the denoise moves, the VAE within "
                      f"{drift:.3f} s", loc="left", fontsize=13, pad=10)
    axes[1].set_ylabel("Seconds, log scale")
    axes[1].legend(loc="lower left", framealpha=0.92)
    axes[1].margins(y=0.22)

    fig.text(0.5, 0.055,
             "Bars: one process per arm, the same binary, H3_DIT_ATTENTION_MMA the only difference, so each pair is its\n"
             "own control. The scalar arm's render is byte-identical to the campaign's gate render, 69816f7f..., 1,769,313\n"
             "bytes; the tensor arm's differs as it must and measures PSNR 42.75 dB luma with SSIM 0.9868 against it.\n"
             "The denoise is 96.3 percent of the scalar step, so the wall keeps only 2.97x of the kernel's 29.4x.",
             ha="center", va="center", fontsize=9, color="#4b5563", linespacing=1.6)
    fig.subplots_adjust(top=0.885, bottom=0.165, left=0.1, right=0.97, hspace=0.34)
    target = DATA.parent / "h3-jobA-2026-10-27.png"
    fig.savefig(target, dpi=180)
    plt.close(fig)
    print(f"wrote {target}")


if __name__ == "__main__":
    plot()
