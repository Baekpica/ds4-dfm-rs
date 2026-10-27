#!/usr/bin/env python3
"""Plot the 512x512, 20-step pair both ways; requires matplotlib.

Data: docs/benchmarks/h3-512-2026-10-27/h3-512-both-arms.csv, one row per arm
per phase, one observable per scope (hq512-rain, hq512-earth). Styling follows
plot-h3-jobA.py: the headline pair on top with values on the bars, the full
phase breakdown below on a log axis so it is visible that only the denoise
moves, and a footnote stating the control and the quality figures.
"""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DATA = Path(__file__).with_name("h3-512-2026-10-27")
COLORS = {"scalar": "#687789", "tensor": "#087f8c"}
LABELS = {"scalar": "Scalar kernel", "tensor": "Tensor core"}
HEADLINE = (("DENOISE_20steps", "Denoise (20 steps)"), ("wall", "Wall, whole job"))
PHASES = (("TEXT", "Text"), ("LOAD", "Load"), ("DENOISE_20steps", "Denoise"),
          ("AUDIO", "Audio"), ("FRAMES", "Frames (VAE)"), ("VIDEO", "Video"))


def load():
    rows = {}
    with (DATA / "h3-512-both-arms.csv").open() as source:
        for row in csv.DictReader(source):
            if not row["scope"].startswith("hq512-"):
                continue
            try:
                value = float(row["value"])
            except ValueError:
                continue    # the hashes share this file and are not numbers
            arm = row["arm"] if row["arm"] in LABELS else "pair"
            rows[(row["scope"], arm, row["metric"])] = value
    return rows


def bars(ax, entries, rows, scope, log=False, unit="s"):
    width = 0.38
    for index, (metric, heading) in enumerate(entries):
        for offset, arm in ((-0.5, "scalar"), (0.5, "tensor")):
            value = rows[(scope, arm, metric)]
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


def plot(scope="hq512-rain", subject="Rain, 512x512, 56 frames, 20 steps",
         suffix=""):
    rows = load()
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.facecolor": "white"})
    fig, axes = plt.subplots(2, 1, figsize=(11, 8.5))
    fig.suptitle("MiniMax-H3 fully denoised · scalar vs tensor core", fontsize=18,
                 fontweight="bold", y=0.985)
    fig.text(0.5, 0.945,
             f"One DGX Spark / GB10 · {subject} · seed 42 / 1 · 2026-10-27",
             ha="center", color="#4b5563")

    denoise = (rows[(scope, "scalar", "DENOISE_20steps")] /
               rows[(scope, "tensor", "DENOISE_20steps")])
    wall = rows[(scope, "scalar", "wall")] / rows[(scope, "tensor", "wall")]
    bars(axes[0], HEADLINE, rows, scope)
    axes[0].set_title(f"Headline · denoise {denoise:.2f}x shorter, wall {wall:.2f}x",
                      loc="left", fontsize=13, pad=10)
    axes[0].set_ylabel("Seconds")
    axes[0].legend(loc="upper right", framealpha=0.92)
    axes[0].margins(y=0.18)

    bars(axes[1], PHASES, rows, scope, log=True)
    drift = abs(rows[(scope, "scalar", "FRAMES")] - rows[(scope, "tensor", "FRAMES")])
    axes[1].set_title("Phase walls · only the denoise moves, the VAE within "
                      f"{drift:.3f} s", loc="left", fontsize=13, pad=10)
    axes[1].set_ylabel("Seconds, log scale")
    axes[1].legend(loc="lower left", framealpha=0.92)
    axes[1].margins(y=0.22)

    psnr = rows[(scope, "pair", "psnr_y")]
    ssim = rows[(scope, "pair", "ssim_all")]
    share = rows[(scope, "scalar", "attention_share_of_step")]
    fig.text(0.5, 0.055,
             "Bars: one process per arm, the same binary, H3_DIT_ATTENTION_MMA the only difference, so each pair is its own control.\n"
             "At 20 steps the two arms diverge more than at 2, which is the expected direction: a per-step difference compounds through\n"
             f"the denoise. This subject: luma PSNR {psnr:.2f} dB, SSIM {ssim:.4f}. Job A at 2 steps: 42.75 dB, 0.9868.\n"
             f"The attention is {100 * share:.1f} percent of the scalar step at this shape, against 96.3 percent at 18,432 tokens.",
             ha="center", va="center", fontsize=9, color="#4b5563", linespacing=1.6)
    fig.subplots_adjust(top=0.885, bottom=0.165, left=0.1, right=0.97, hspace=0.34)
    target = DATA.parent / f"h3-512-2026-10-27{suffix}.png"
    fig.savefig(target, dpi=180)
    plt.close(fig)
    print(f"wrote {target}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 2:
        plot(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "")
    else:
        plot()
