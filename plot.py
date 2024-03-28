import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


x = ["FP16", "AWQ", "LLM.int4()", "$\\text{L}^2\\text{QER}$"]
tps_70b = [0, 460.15, 850.97, 1442.63]

tps_30b = [0, 1556.07, 1654.59, 2106.9]

tps_13b = [3511.8, 3805.80, 4131.15, 4676.55]

tps_7b = [6929.12, 8155.12, 8070.56, 8449.22]

fp16 = [tps_7b[0], tps_13b[0], tps_30b[0], tps_70b[0]]
awq = [tps_7b[1], tps_13b[1], tps_30b[1], tps_70b[1]]
llm_int4 = [tps_7b[2], tps_13b[2], tps_30b[2], tps_70b[2]]
l2qer_fp8fp16 = [tps_7b[3], tps_13b[3], tps_30b[3], tps_70b[3]]

model_size = ["7b", "13b", "30b", "70b"]

width = 0.6
multiplier = 0
font_size = 12


f, axes = plt.subplots(1, 4, figsize=(16, 4), dpi=300)

for i, ax in enumerate(axes):
    ax.set_adjustable("box")
    y = [fp16[i], awq[i], llm_int4[i], l2qer_fp8fp16[i]]
    y_lim = (0, round(max(y) * 1.2))
    ax.bar(x, y, width=width)
    if y[0] == 0:
        # normalize to awq and round to 2 decimal
        normalized = [
            round((fp16[i] / awq[i]), 2),
            1,
            round((llm_int4[i] / awq[i]), 2),
            round((l2qer_fp8fp16[i] / awq[i]), 2),
        ]
    else:
        # normalize to fp16 and round to 1 decimal
        normalized = [
            1,
            round((awq[i] / fp16[i]), 2),
            round((llm_int4[i] / fp16[i]), 2),
            round((l2qer_fp8fp16[i] / fp16[i]), 2),
        ]

    normalized_str = [str(x) + "x" if x > 0.01 else "NA\n(OOM)" for x in normalized]
    # set text on top of bars in bold
    for j, v in enumerate([fp16[i], awq[i], llm_int4[i], l2qer_fp8fp16[i]]):
        ax.text(
            j,
            v + 30,
            normalized_str[j],
            ha="center",
            va="bottom",
            fontsize=font_size * 0.8,
            fontweight="bold",
        )
    ax.set_ylim(y_lim)
    title = (
        f"LLAMA-{model_size[i]} (RTX 6000 Ada)"
        if i < 3
        else f"LLAMA-{model_size[i]} (H100)"
    )
    ax.set_title(title, fontsize=font_size)
    ax.set_ylabel("Tokens per second", fontsize=font_size)
    ax.set_xticklabels(x, rotation=45)
    ax.grid(axis="y")

# f.tight_layout()
plt.subplots_adjust(bottom=0.2, wspace=0.5)
plt.savefig("gpu_perf.png")
