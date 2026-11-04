#!/usr/bin/env python3
"""Make the pilot quality/latency figure from score-summary.json."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

root = Path(__file__).resolve().parent
rows = json.loads((root / "score-summary.json").read_text())["runs"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharex=True, sharey=True)
colors = {"tiny.en": "#177E89", "base.en": "#B85B32"}
markers = {4: "o", 8: "s"}
for ax, key, title in zip(axes, ["longyaal_cu_s", "longyaal_ca_s"],
                          ["Computation excluded", "Computation included"]):
    for row in rows:
        x, y = row[key], row["resegmented_chrf2"]
        ax.scatter(x, y, color=colors[row["asr"]], marker=markers[row["chunk_s"]],
                   s=90, edgecolor="white", linewidth=0.7, zorder=3)
        name = f"{row['asr'].replace('.en', '')}, {row['chunk_s']} s"
        offset = (8, -15) if row["asr"] == "tiny.en" else (8, 8)
        ax.annotate(name, (x, y), xytext=offset, textcoords="offset points", fontsize=10)
    ax.set_title(title, loc="left", fontsize=13, pad=12)
    ax.set_xlim(0, 8)
    ax.set_ylim(51, 59)
    ax.set_xticks(range(0, 9, 2))
    ax.set_yticks([52, 54, 56, 58])
    ax.set_xlabel("LongYAAL (seconds; lower is better)", labelpad=10)
    ax.grid(alpha=0.2, linewidth=0.6)
axes[0].set_ylabel("Reference-resegmented chrF2 (higher is better)", labelpad=10)
fig.suptitle("Including computation changes the pilot quality–latency trade-off", x=0.08,
             ha="left", fontsize=15, fontweight="bold", y=0.96)
fig.text(0.08, 0.055, "MCIF JhbtCwcsWY · Raspberry Pi 5 · one run per configuration · 39 quality segments",
         color="#454545", fontsize=10)
fig.text(0.08, 0.018, "OmniSTEval 0.1.10 · LongYAAL averages 38 contributing segments; final-tail delay is reported separately.",
         color="#454545", fontsize=9)
fig.subplots_adjust(left=0.08, right=0.98, bottom=0.20, top=0.81, wspace=0.12)
fig.savefig(root / "Pilot-quality-latency.png", dpi=180)
fig.savefig(root / "Pilot-quality-latency.svg")
