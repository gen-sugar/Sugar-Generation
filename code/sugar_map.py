# -*- coding: utf-8 -*-
"""
Sugar consumption scenario maps (2030 / 2035 / 2040 / 2045)
Averages 5 years of FAOSTAT 'Sugar (Raw Equivalent), Food supply quantity
(kg/capita/yr)' data per country to build a 2030 baseline map
2035/2040/2045 values using a rule-based scenario
"""

import os
import unicodedata

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np
import pandas as pd
import geopandas as gpd
import cartopy.crs as ccrs
from matplotlib.path import Path
from matplotlib.patches import PathPatch

# Paths

BASE_DIR = Path(__file__).resolve().parent.parent

CSV_PATH = BASE_DIR / "data" / "FAOSTAT_sugar.csv"
SHP_PATH = BASE_DIR / "data" / "ne_50m_admin_0_countries.shp"

OUT_DIR = BASE_DIR / "output"
OUT_DIR.mkdir(exist_ok=True)


def strip_accents(text):
    """Normalize a string to plain ASCII (strips accents, e.g. the
    o-with-circumflex in 'Ivoire' -> plain 'o'). Used so that matching does
    not depend on this .py file containing any literal accented characters."""
    if not isinstance(text, str):
        return text
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))


# Load FAOSTAT data 
df = pd.read_csv(CSV_PATH)

# Sugar (Raw Equivalent) Food supply quantity (kg/capita/yr)
df = df[
    (df["Item"] == "Sugar (Raw Equivalent)")
    & (df["Element"] == "Food supply quantity (kg/capita/yr)")
].copy()

df = df[df["Area"] != "China"]

master_mapping = {
    "China, mainland": "China",
    "China, Hong Kong SAR": "Hong Kong",
    "China, Macao SAR": "Macao",
    "China, Taiwan Province of": "Taiwan",
    "Democratic Republic of the Congo": "Dem. Rep. Congo",
    "Lao People's Democratic Republic": "Laos",
    "Republic of Korea": "South Korea",
    "United Republic of Tanzania": "Tanzania",
}


# Mapping

faostat_name_map = {
    "Antigua and Barbuda": "Antigua and Barb.",
    "Bolivia (Plurinational State of)": "Bolivia",
    "Bosnia and Herzegovina": "Bosnia and Herz.",
    "China, mainland": "China",
    "China, Hong Kong SAR": "Hong Kong",
    "China, Macao SAR": "Macao",
    "China, Taiwan Province of": "Taiwan",
    "Democratic Republic of the Congo": "Dem. Rep. Congo",
    "Dominican Republic": "Dominican Rep.",
    "Eswatini": "eSwatini",
    "French Polynesia": "Fr. Polynesia",
    "Iran (Islamic Republic of)": "Iran",
    "Lao People's Democratic Republic": "Laos",
    "Marshall Islands": "Marshall Is.",
    "Micronesia (Federated States of)": "Micronesia",
    "Naoero": "Nauru",
    "Netherlands (Kingdom of the)": "Netherlands",
    "Republic of Korea": "South Korea",
    "Republic of Moldova": "Moldova",
    "Russian Federation": "Russia",
    "Saint Kitts and Nevis": "St. Kitts and Nevis",
    "Sao Tome and Principe": "Sao Tome and Principe",
    "Solomon Islands": "Solomon Is.",
    "Syrian Arab Republic": "Syria",
    "Turkiye": "Turkey",  # FAOSTAT's raw value has an accented u; the
                          # strip_accents() step below normalizes it to
                          # "Turkiye" (plain ASCII) before this lookup runs
    "United Kingdom of Great Britain and Northern Ireland": "United Kingdom",
    "United Republic of Tanzania": "Tanzania",
    "Venezuela (Bolivarian Republic of)": "Venezuela",
    "Viet Nam": "Vietnam",
    # Tiny island states with no polygon in the 1:50m map
}

NAME_MAP = {**master_mapping, **faostat_name_map}

# Normalize text, key values
NAME_MAP_ASCII = {strip_accents(k): strip_accents(v) for k, v in NAME_MAP.items()}
df["area_ascii"] = df["Area"].map(strip_accents)
df["map_name"] = df["area_ascii"].map(lambda x: NAME_MAP_ASCII.get(x, x))


# 5-year average per country
baseline = (
    df.groupby(["Area", "map_name"], as_index=False)["Value"]
    .mean()
    .rename(columns={"Value": "y2030"})
)

# Scenario
# Major cane-sugar producers (Brazil etc.) see a temporary consumption surge as the sugar crisis unfolds.
# Early-adopter economies (US, China, Korea, Japan, western Europe...) shift to sweeteners
# Everyone else declines more gradually.
# By 2040-2045 the whole world converges toward near-zero consumption.


SURGE_COUNTRIES = {
    "Brazil", "Mexico", "India", "Thailand", "Indonesia", "Pakistan",
    "Guatemala", "Cuba", "Colombia", "Philippines",
}
FAST_TRANSITION_COUNTRIES = {
    "China, mainland", "United States of America", "Republic of Korea",
    "Japan", "Germany", "France", "United Kingdom of Great Britain and Northern Ireland",
    "Canada", "Australia",
}


def scenario_multiplier(area, year):
    """Multiplier relative to the 2030 baseline (1.0), by country group."""
    if area in SURGE_COUNTRIES:
        return {2035: 1.35, 2040: 0.55, 2045: 0.10}[year]
    elif area in FAST_TRANSITION_COUNTRIES:
        return {2035: 0.55, 2040: 0.15, 2045: 0.03}[year]
    else:
        return {2035: 0.85, 2040: 0.40, 2045: 0.08}[year]


for year in (2035, 2040, 2045):
    baseline[f"y{year}"] = baseline.apply(
        lambda r: r["y2030"] * scenario_multiplier(r["Area"], year), axis=1
    )

# Merge with the shapefile

world = gpd.read_file(SHP_PATH)
world["NAME_ascii"] = world["NAME"].map(strip_accents)
world = world[world["NAME_ascii"] != "Antarctica"].copy()

merged = world.merge(baseline, left_on="NAME_ascii", right_on="map_name", how="left")

matched = merged["y2030"].notna().sum()
print(f"[match] {matched} of {len(world)} map polygons matched to data")

unmatched_faostat = set(baseline["Area"]) - set(
    baseline.loc[baseline["map_name"].isin(world["NAME_ascii"]), "Area"]
)
if unmatched_faostat:
    print("[unmatched countries -- not shown on the map]:")
    for a in sorted(unmatched_faostat):
        print("   -", a)


china_row = merged[merged["NAME_ascii"] == "China"]
if not china_row.empty:
    print(f"[check] China (mainland) 2030 average = {china_row['y2030'].values[0]:.2f} "
          f"kg/capita/yr (low value -> light shade, but the match itself is correct)")

bins = [0, 12, 24, 36, 48, 60, np.inf]
bin_labels = ["12", "24", "36", "48", "60"]
n_bins = len(bins) - 1
gray_colors = plt.cm.Greys(np.linspace(0.08, 0.95, n_bins))
cmap = ListedColormap(gray_colors)
cmap.set_bad(color="white")
norm = BoundaryNorm(bins, cmap.N)

# Color bar
def draw_arrow_colorbar(fig, ax, cmap, bin_labels):
    bbox = ax.get_position()
    bar_h = 0.032
    bar_y = bbox.y0 - 0.045
    total_w = bbox.width * 0.7
    bar_x0 = bbox.x0 + (bbox.width - total_w) / 2

    n = cmap.N
    tri_frac = 0.6 
    n_units = n + 2 * tri_frac
    box_w = total_w / n_units
    tri_w = box_w * tri_frac

    bar_ax = fig.add_axes([bar_x0, bar_y, total_w, bar_h])
    bar_ax.set_xlim(0, total_w)
    bar_ax.set_ylim(0, 1)
    bar_ax.axis("off")

    left_tri = plt.Polygon(
        [[tri_w, 0], [tri_w, 1], [0, 0.5]],
        closed=True, facecolor=cmap(0), edgecolor="black", linewidth=0.8,
    )
    bar_ax.add_patch(left_tri)

    for i in range(n):
        x0 = tri_w + i * box_w
        rect = plt.Rectangle(
            (x0, 0), box_w, 1,
            facecolor=cmap(i), edgecolor="black", linewidth=0.8,
        )
        bar_ax.add_patch(rect)

    x_end = tri_w + n * box_w
    right_tri = plt.Polygon(
        [[x_end, 0], [x_end, 1], [x_end + tri_w, 0.5]],
        closed=True, facecolor=cmap(n - 1), edgecolor="black", linewidth=0.8,
    )
    bar_ax.add_patch(right_tri)


    for i, label in enumerate(bin_labels):
        x = tri_w + (i + 1) * box_w
        txt = bar_ax.text(x, 0.5, label, ha="center", va="center", fontsize=5,
                           zorder=10, color="black")
        txt.set_path_effects([pe.withStroke(linewidth=2.5, foreground="white")])


# Mapping
years = [2030, 2035, 2040, 2045]

map_crs = ccrs.Robinson(central_longitude=0)
data_crs = ccrs.PlateCarree()

for year in years:
    col = f"y{year}"

    fig = plt.figure(figsize=(8, 4.6))

    ax = fig.add_axes(
        [0.02, 0.15, 0.96, 0.80],
        projection=map_crs
    )

    ax.set_global()

    for i in range(n_bins):

        lower = bins[i]
        upper = bins[i + 1]

        if np.isinf(upper):
            subset = merged[
                (merged[col] >= lower) &
                (merged[col].notna())
            ]
        else:
            subset = merged[
                (merged[col] >= lower) &
                (merged[col] < upper) &
                (merged[col].notna())
            ]

        if not subset.empty:
            ax.add_geometries(
                subset.geometry,
                crs=data_crs,
                facecolor=cmap(i),
                edgecolor="black",
                linewidth=0.35,
                zorder=2
            )

    
    # Countries without data - white
    missing = merged[merged[col].isna()]

    if not missing.empty:
        ax.add_geometries(
            missing.geometry,
            crs=data_crs,
            facecolor="white",
            edgecolor="black",
            linewidth=0.35,
            zorder=1
        )

    # boundary line

    ax.spines["geo"].set_visible(True)
    ax.spines["geo"].set_linewidth(0.8)
    ax.spines["geo"].set_edgecolor("black")

    ax.set_xticks([])
    ax.set_yticks([])

    # color bar

    draw_arrow_colorbar(
        fig,
        ax,
        cmap,
        bin_labels
    )


    # Save

    out_path = os.path.join(
        OUT_DIR,
        f"sugar_map_{year}.png"
    )

    fig.savefig(
        out_path,
        dpi=300,
        facecolor="white"
    )

    plt.close(fig)

    print(
        f"[done] saved {year} map: {out_path}"
    )

# scenario values table 

baseline_out = baseline[["Area", "y2030", "y2035", "y2040", "y2045"]].sort_values("Area")
baseline_out.to_csv(os.path.join(OUT_DIR, "sugar_scenario_values.csv"), index=False)
print(f"[done] saved value table: {os.path.join(OUT_DIR, 'sugar_scenario_values.csv')}")