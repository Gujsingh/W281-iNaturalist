"""Size / resolution / aspect-ratio variation across categories.
Usage: python variation_analysis.py sample_images/metadata.csv outdir
Uses only metadata.csv (width/height are in it), so no images are needed.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

meta, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
d = pd.read_csv(meta, index_col=0)
d["megapixels"] = d.width * d.height / 1e6
d["aspect"] = d.width / d.height
d["long_side"] = d[["width", "height"]].max(axis=1)
d["orientation"] = np.where(d.width > d.height, "landscape",
                   np.where(d.width < d.height, "portrait", "square"))

order = d.groupby("supercategory").aspect.median().sort_values().index.tolist()

summary = d.groupby("supercategory").agg(
    n=("width", "size"),
    species=("species", "nunique"),
    width_mean=("width", "mean"), height_mean=("height", "mean"),
    mp_mean=("megapixels", "mean"), mp_std=("megapixels", "std"),
    aspect_median=("aspect", "median"), aspect_std=("aspect", "std"),
    pct_long_side_500=("long_side", lambda s: (s == 500).mean() * 100),
    min_long_side=("long_side", "min"),
).round(3)
orient = pd.crosstab(d.supercategory, d.orientation, normalize="index").mul(100).round(1)
summary = summary.join(orient)
summary.to_csv(out / "variation_summary.csv")
print(summary.to_string())

fig, ax = plt.subplots(2, 2, figsize=(14, 9))
d.boxplot(column="aspect", by="supercategory", ax=ax[0, 0], rot=60)
ax[0, 0].set_title("Aspect ratio (w/h)"); ax[0, 0].axhline(1, color="gray", ls="--")
d.boxplot(column="megapixels", by="supercategory", ax=ax[0, 1], rot=60)
ax[0, 1].set_title("Megapixels")
orient.loc[order, ["landscape", "square", "portrait"]].plot.barh(stacked=True, ax=ax[1, 0])
ax[1, 0].set_title("Orientation mix (%)")
for c in order:
    ax[1, 1].scatter(d[d.supercategory == c].width, d[d.supercategory == c].height, s=4, alpha=.4, label=c)
ax[1, 1].set_xlabel("width"); ax[1, 1].set_ylabel("height"); ax[1, 1].set_title("Width vs height")
ax[1, 1].legend(fontsize=6, markerscale=3)
fig.suptitle("")
fig.tight_layout()
fig.savefig(out / "variation_overview.png", dpi=150)
