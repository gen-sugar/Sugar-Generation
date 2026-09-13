# -*- coding: utf-8 -*-
"""
Sweetener consumption map (2030)
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



# Paths

from pathlib import Path

# Project root
BASE_DIR = Path(__file__).resolve().parent.parent

CSV_PATH = BASE_DIR / "data" / "FAOSTAT_sweetener.csv"

SHP_PATH = BASE_DIR / "data" / "ne_50m_admin_0_countries.shp"

OUT_DIR = BASE_DIR / "output"
OUT_DIR.mkdir(exist_ok=True)



# FAOSTAT load


ITEM_NAME = "Sweeteners, Other"


ELEMENT_NAME = "Food supply quantity (kg/capita/yr)"


def strip_accents(text):
    """Normalize text to plain ASCII."""

    if not isinstance(text, str):
        return text

    nfkd = unicodedata.normalize("NFKD", text)

    return "".join(
        ch for ch in nfkd
        if not unicodedata.combining(ch)
    )



# Load FAOSTAT data

df = pd.read_csv(CSV_PATH)


print("[FAOSTAT] Items found:")
print(df["Item"].dropna().unique())



df = df[
    (df["Item"] == ITEM_NAME)
    & (df["Element"] == ELEMENT_NAME)
].copy()


if df.empty:
    raise ValueError(
        f'No data found for Item="{ITEM_NAME}" '
        f'and Element="{ELEMENT_NAME}". '
        f'Check the exact Item name printed above.'
    )

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


faostat_name_map = {

    "Antigua and Barbuda":
        "Antigua and Barb.",

    "Bolivia (Plurinational State of)":
        "Bolivia",

    "Bosnia and Herzegovina":
        "Bosnia and Herz.",

    "China, mainland":
        "China",

    "China, Hong Kong SAR":
        "Hong Kong",

    "China, Macao SAR":
        "Macao",

    "China, Taiwan Province of":
        "Taiwan",

    "Democratic Republic of the Congo":
        "Dem. Rep. Congo",

    "Dominican Republic":
        "Dominican Rep.",

    "Eswatini":
        "eSwatini",

    "French Polynesia":
        "Fr. Polynesia",

    "Iran (Islamic Republic of)":
        "Iran",

    "Lao People's Democratic Republic":
        "Laos",

    "Marshall Islands":
        "Marshall Is.",

    "Micronesia (Federated States of)":
        "Micronesia",

    "Naoero":
        "Nauru",

    "Netherlands (Kingdom of the)":
        "Netherlands",

    "Republic of Korea":
        "South Korea",

    "Republic of Moldova":
        "Moldova",

    "Russian Federation":
        "Russia",

    "Saint Kitts and Nevis":
        "St. Kitts and Nevis",

    "Sao Tome and Principe":
        "Sao Tome and Principe",

    "Solomon Islands":
        "Solomon Is.",

    "Syrian Arab Republic":
        "Syria",

    "Turkiye":
        "Turkey",

    "United Kingdom of Great Britain and Northern Ireland":
        "United Kingdom",

    "United Republic of Tanzania":
        "Tanzania",

    "Venezuela (Bolivarian Republic of)":
        "Venezuela",

    "Viet Nam":
        "Vietnam",
}


NAME_MAP = {
    **master_mapping,
    **faostat_name_map
}


# Normalize mapping to ASCII
NAME_MAP_ASCII = {
    strip_accents(k): strip_accents(v)
    for k, v in NAME_MAP.items()
}


df["area_ascii"] = df["Area"].map(strip_accents)

df["map_name"] = df["area_ascii"].map(
    lambda x: NAME_MAP_ASCII.get(x, x)
)


# 5-year average -> 2030


baseline = (
    df.groupby(
        ["Area", "map_name"],
        as_index=False
    )["Value"]
    .mean()
    .rename(
        columns={"Value": "y2030"}
    )
)


print(
    f"[data] {len(baseline)} countries/areas "
    "have 2030 values"
)


# Load world shapefile


world = gpd.read_file(SHP_PATH)

world["NAME_ascii"] = world["NAME"].map(
    strip_accents
)


merged = world.merge(
    baseline,
    left_on="NAME_ascii",
    right_on="map_name",
    how="left"
)


matched = merged["y2030"].notna().sum()

print(
    f"[match] {matched} of {len(world)} "
    "map polygons matched to data"
)


bins = [
    0,
    12,
    24,
    36,
    48,
    60,
    np.inf
]

bin_labels = [
    "12",
    "24",
    "36",
    "48",
    "60"
]

n_bins = len(bins) - 1


gray_colors = plt.cm.Greys(
    np.linspace(
        0.08,
        0.95,
        n_bins
    )
)

cmap = ListedColormap(
    gray_colors
)

cmap.set_bad(
    color="white"
)

norm = BoundaryNorm(
    bins,
    cmap.N
)


# color bar


def draw_arrow_colorbar(
    fig,
    ax,
    cmap,
    bin_labels
):

    bbox = ax.get_position()

    bar_h = 0.032

    bar_y = bbox.y0 - 0.045

    total_w = bbox.width * 0.7

    bar_x0 = (
        bbox.x0
        + (bbox.width - total_w) / 2
    )

    n = cmap.N

    tri_frac = 0.6

    n_units = (
        n
        + 2 * tri_frac
    )

    box_w = (
        total_w
        / n_units
    )

    tri_w = (
        box_w
        * tri_frac
    )


    bar_ax = fig.add_axes(
        [
            bar_x0,
            bar_y,
            total_w,
            bar_h
        ]
    )

    bar_ax.set_xlim(
        0,
        total_w
    )

    bar_ax.set_ylim(
        0,
        1
    )

    bar_ax.axis("off")


    # Left arrow
    left_tri = plt.Polygon(
        [
            [tri_w, 0],
            [tri_w, 1],
            [0, 0.5]
        ],
        closed=True,
        facecolor=cmap(0),
        edgecolor="black",
        linewidth=0.8
    )

    bar_ax.add_patch(
        left_tri
    )


    # Middle boxes
    for i in range(n):

        x0 = (
            tri_w
            + i * box_w
        )

        rect = plt.Rectangle(
            (x0, 0),
            box_w,
            1,
            facecolor=cmap(i),
            edgecolor="black",
            linewidth=0.8
        )

        bar_ax.add_patch(
            rect
        )


    # Right arrow
    x_end = (
        tri_w
        + n * box_w
    )

    right_tri = plt.Polygon(
        [
            [x_end, 0],
            [x_end, 1],
            [x_end + tri_w, 0.5]
        ],
        closed=True,
        facecolor=cmap(n - 1),
        edgecolor="black",
        linewidth=0.8
    )

    bar_ax.add_patch(
        right_tri
    )


    # Labels
    for i, label in enumerate(
        bin_labels
    ):

        x = (
            tri_w
            + (i + 1) * box_w
        )

        txt = bar_ax.text(
            x,
            0.5,
            label,
            ha="center",
            va="center",
            fontsize=7,
            zorder=10,
            color="black"
        )

        txt.set_path_effects(
            [
                pe.withStroke(
                    linewidth=1.5,
                    foreground="white"
                )
            ]
        )



# 2030 Sweetener map


year = 2030
col = "y2030"

map_crs = ccrs.Robinson(central_longitude=0)
data_crs = ccrs.PlateCarree()

fig = plt.figure(figsize=(8, 4.6))

ax = fig.add_axes(
    [0.02, 0.15, 0.96, 0.80],
    projection=map_crs
)

ax.set_global()

# Plot each value class
for i in range(n_bins):

    lower = bins[i]
    upper = bins[i + 1]

    if np.isinf(upper):
        subset = merged[
            (merged[col] >= lower) &
            merged[col].notna()
        ]
    else:
        subset = merged[
            (merged[col] >= lower) &
            (merged[col] < upper) &
            merged[col].notna()
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

# Missing data = white
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

# Robinson outer boundary
ax.spines["geo"].set_visible(True)
ax.spines["geo"].set_linewidth(0.8)
ax.spines["geo"].set_edgecolor("black")

ax.set_xticks([])
ax.set_yticks([])

# Color bar
draw_arrow_colorbar(
    fig,
    ax,
    cmap,
    bin_labels
)

# ? Sweetener ?? ???
out_path = os.path.join(
    OUT_DIR,
    "sweetener_map_2030.png"
)

fig.savefig(
    out_path,
    dpi=300,
    facecolor="white"
)

plt.close(fig)

print(f"[done] saved Sweetener 2030 map: {out_path}")


# Save 2030 values


baseline_out = (
    baseline[
        ["Area", "y2030"]
    ]
    .sort_values("Area")
)

table_path = os.path.join(
    OUT_DIR,
    "sweetener_2030_values.csv"
)

baseline_out.to_csv(
    table_path,
    index=False
)

print(f"[done] saved Sweetener 2030 values: {table_path}")



for i in range(n_bins):

    lower = bins[i]
    upper = bins[i + 1]


    if np.isinf(upper):

        subset = merged[
            (merged[col] >= lower)
            & merged[col].notna()
        ]

    else:

        subset = merged[
            (merged[col] >= lower)
            & (merged[col] < upper)
            & merged[col].notna()
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


missing = merged[
    merged[col].isna()
]


if not missing.empty:

    ax.add_geometries(
        missing.geometry,
        crs=data_crs,
        facecolor="white",
        edgecolor="black",
        linewidth=0.35,
        zorder=1
    )



# boundary


ax.spines["geo"].set_visible(True)

ax.spines["geo"].set_linewidth(
    0.8
)

ax.spines["geo"].set_edgecolor(
    "black"
)


# Remove normal axes
ax.set_xticks([])
ax.set_yticks([])


draw_arrow_colorbar(
    fig,
    ax,
    cmap,
    bin_labels
)


# Save map


out_path = os.path.join(
    OUT_DIR,
    "sweetener_map_2030.png"
)


fig.savefig(
    out_path,
    dpi=300,
    facecolor="white"
)


plt.close(fig)


print(
    f"[done] saved 2030 map: {out_path}"
)


# Save values


baseline_out = (
    baseline[
        [
            "Area",
            "y2030"
        ]
    ]
    .sort_values("Area")
)


table_path = os.path.join(
    OUT_DIR,
    "sweetener_2030_values.csv"
)


baseline_out.to_csv(
    table_path,
    index=False
)


print(
    f"[done] saved value table: {table_path}"
)