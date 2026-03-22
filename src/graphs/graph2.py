import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

from config import DPI, FORMAT, OUTPUT_DIR, ROOT_DIR
from style import BLUE, GRAY, GREEN, WHITE, YELLOW, apply_global_style


def get_mode(series, is_numeric=False):
    """Calculate the mode of a series, ignoring empty strings and NaNs, normalized."""
    m = series.dropna()
    m = m[m.astype(str).str.strip() != ""]
    if m.empty:
        return "N/A"

    if is_numeric:
        return m.mode().iloc[0]

    # Normalize categories to avoid "blue" vs "Blue"
    m = m.astype(str).str.strip().str.lower()
    mode_val = m.mode().iloc[0]
    return mode_val.capitalize()


def run():
    logger.info("Generating Graph 2: Multidimensional Whaler Profiles (Categorized)")

    # Load data
    data_path = ROOT_DIR / "data" / "raw" / "original_crew_dataset.csv"
    try:
        df = pd.read_csv(data_path, engine="python", on_bad_lines="skip")
    except Exception as e:
        logger.error(f"Failed to read dataset: {e}")
        return

    # Preprocessing
    df["height_feet"] = pd.to_numeric(df["height_feet"], errors="coerce")
    df["height_inches"] = pd.to_numeric(df["height_inches"], errors="coerce")
    df["height_cm"] = (df["height_feet"] * 30.48) + (df["height_inches"] * 2.54)
    df["origin"] = df["birthplace"].fillna(df["res_city"])

    # Define target cities with color groupings
    targets = {
        # Centers (Blue)
        "New Bedford": {"filter": lambda x: str(x) == "New Bedford", "color": BLUE},
        "New London": {"filter": lambda x: str(x) == "New London", "color": BLUE},
        "New York": {"filter": lambda x: str(x) == "New York", "color": BLUE},
        # Regional Travelers (Green)
        "Albany": {"filter": lambda x: str(x) == "Albany", "color": GREEN},
        "Philadelphia": {"filter": lambda x: str(x) == "Philadelphia", "color": GREEN},
        "Rochester": {"filter": lambda x: str(x) == "Rochester", "color": GREEN},
        # Foreigners (Yellow)
        "Germany": {"filter": lambda x: str(x) == "Germany", "color": YELLOW},
        "Azores": {
            "filter": lambda x: any(
                island.lower() in str(x).lower()
                for island in ["Azores", "Fayal", "Pico", "Flores", "Sao Jorge"]
            ),
            "color": YELLOW,
        },
    }

    features = {
        "Рост (см)": "height_cm",
        "Цвет кожи": "skin",
        "Цвет волос": "hair",
        "Цвет глаз": "eye",
        "Должность": "rank",
    }

    # 1. Identify Top 20 ports for background context
    valid_origins = df[
        ~df["origin"]
        .fillna("")
        .str.lower()
        .isin(["", "unknown", "not stated", "-", "`"])
    ]
    top_20_origins = valid_origins["origin"].value_counts().head(20).index.tolist()

    # 2. Calculate profiles for all top 20 + targets
    all_profiles = []

    # Track categories for axes
    cat_values = {label: set() for label in features if label != "Рост (см)"}
    heights = []

    # First, process target profiles
    for name, config in targets.items():
        subset = df[df["origin"].apply(config["filter"])].copy()
        profile = {
            "name": name,
            "color": config["color"],
            "is_target": True,
            "linewidth": 4,
            "alpha": 0.8,
        }
        for label, col in features.items():
            is_num = label == "Рост (см)"
            val = get_mode(subset[col], is_numeric=is_num)
            profile[label] = val
            if is_num:
                if val != "N/A":
                    heights.append(val)
            else:
                if val != "N/A":
                    cat_values[label].add(val)
        all_profiles.append(profile)

    # Second, process background profiles (Top 20 ports not already in targets)
    target_names = list(targets.keys())
    for origin in top_20_origins:
        if origin in target_names:
            continue

        subset = df[df["origin"] == origin].copy()
        profile = {
            "name": origin,
            "color": "#DDDDDD",
            "is_target": False,
            "linewidth": 1.0,
            "alpha": 0.2,
        }
        for label, col in features.items():
            is_num = label == "Рост (см)"
            val = get_mode(subset[col], is_numeric=is_num)
            profile[label] = val
            if is_num:
                if val != "N/A":
                    heights.append(val)
            else:
                if val != "N/A":
                    cat_values[label].add(val)
        all_profiles.append(profile)

    # 3. Prepare axis info
    axes_info = {}
    for label, _col in features.items():
        if label == "Рост (см)":
            h_min = min(heights) if heights else 160
            h_max = max(heights) if heights else 180
            axes_info[label] = {"type": "num", "min": h_min - 2, "max": h_max + 2}
        else:
            # Sort categories for cleaner lines
            sorted_cats = sorted(cat_values[label])
            if not sorted_cats:
                sorted_cats = ["N/A"]
            axes_info[label] = {"type": "cat", "cats": sorted_cats}

    def normalize(val, info):
        if val == "N/A":
            return 0.0  # Bottom for N/A
        if info["type"] == "num":
            return (val - info["min"]) / (info["max"] - info["min"])
        cats = info["cats"]
        if len(cats) <= 1:
            return 0.5
        return cats.index(val) / (len(cats) - 1)

    # Plotting
    apply_global_style()
    fig, ax = plt.subplots(figsize=(20, 12))
    fig.patch.set_facecolor(WHITE)

    feature_labels = list(features.keys())
    x_indices = np.arange(len(feature_labels))

    # Draw axes
    for i in x_indices:
        ax.axvline(i, color=GRAY, linewidth=0.8, alpha=0.3, zorder=1)

    # Plot Background Profiles first
    for p in [prof for prof in all_profiles if not prof["is_target"]]:
        y_vals = [normalize(p[label], axes_info[label]) for label in feature_labels]
        ax.plot(
            x_indices,
            y_vals,
            color=p["color"],
            linewidth=p["linewidth"],
            alpha=p["alpha"],
            zorder=2,
        )

    # Plot Target Profiles on top
    target_profiles = sorted(
        [prof for prof in all_profiles if prof["is_target"]],
        key=lambda p: normalize(p[feature_labels[0]], axes_info[feature_labels[0]]),
    )

    # Stacking logic for labels
    y_orig = [
        normalize(p[feature_labels[0]], axes_info[feature_labels[0]])
        for p in target_profiles
    ]
    y_adj = y_orig.copy()
    min_sep = 0.05
    for i in range(1, len(y_adj)):
        if y_adj[i] < y_adj[i - 1] + min_sep:
            y_adj[i] = y_adj[i - 1] + min_sep
    if y_adj[-1] > 1.0:
        y_adj[-1] = 1.0
        for i in range(len(y_adj) - 2, -1, -1):
            if y_adj[i] > y_adj[i + 1] - min_sep:
                y_adj[i] = y_adj[i + 1] - min_sep

    for idx, p in enumerate(target_profiles):
        y_vals = [normalize(p[label], axes_info[label]) for label in feature_labels]
        y_label = y_adj[idx]

        ax.plot(
            x_indices,
            y_vals,
            color=p["color"],
            linewidth=p["linewidth"],
            alpha=p["alpha"],
            zorder=4,
            marker="o",
            markersize=8,
            markeredgecolor="white",
            markeredgewidth=1,
        )

        ax.text(
            x_indices[0] - 0.2,
            y_label,
            p["name"],
            color=p["color"],
            ha="right",
            va="center",
            fontweight="bold",
            fontsize=20,
        )

    # Axis Labels and Ticks
    ax.set_xticks(x_indices)
    ax.set_xticklabels(feature_labels, fontsize=21, fontweight="bold", color=GRAY)
    ax.set_yticks([])
    ax.set_ylim(-0.05, 1.05)

    # Add tick labels for each vertical axis
    for i, label in enumerate(feature_labels):
        info = axes_info[label]
        if info["type"] == "num":
            ticks = np.linspace(info["min"], info["max"], 5)
            for val in ticks:
                pos = (val - info["min"]) / (info["max"] - info["min"])
                ax.text(
                    i + 0.08,
                    pos,
                    f"{val:.1f}",
                    color=GRAY,
                    fontsize=12,
                    va="center",
                    bbox={
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.6,
                        "pad": 0.5,
                    },
                )
        else:
            cats = info["cats"]
            for j, cat in enumerate(cats):
                pos = j / (len(cats) - 1) if len(cats) > 1 else 0.5
                ax.text(
                    i + 0.08,
                    pos,
                    cat,
                    color=GRAY,
                    fontsize=12,
                    va="center",
                    bbox={
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.6,
                        "pad": 0.5,
                    },
                )

    for spine in ax.spines.values():
        spine.set_visible(False)

    # Titles
    main_title = (
        "Самые частые герои: сравнение профилей выходцев из ключевых центров найма"
    )
    subtitle = (
        "Типичные характеристики (рост, цвет кожи, глаз, волос) и должности на судне"
    )

    fig.suptitle(
        main_title,
        fontsize=33,
        fontweight="bold",
        x=0.5,
        y=0.96,
        ha="center",
        va="top",
        color=GRAY,
    )
    plt.text(
        0.5,
        0.91,
        subtitle,
        fontsize=22,
        ha="center",
        va="center",
        transform=fig.transFigure,
        color=GRAY,
    )

    footer_text = (
        "Источник: Исторические данные об экипажах китобойных судов (XIX - начало XX века). "
        "Серым цветом показаны профили ТОП-20 портов для контекста."
    )
    plt.text(
        0.02,
        0.02,
        footer_text,
        fontsize=15,
        ha="left",
        transform=fig.transFigure,
        color=GRAY,
        style="italic",
    )

    plt.tight_layout(rect=(0.1, 0.05, 0.95, 0.88))

    output_path = OUTPUT_DIR / f"graph2.{FORMAT}"
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    logger.success(f"Graph 2 (Categorized) saved to {output_path}")


if __name__ == "__main__":
    run()
