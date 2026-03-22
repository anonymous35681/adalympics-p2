import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

from config import DPI, FORMAT, OUTPUT_DIR, ROOT_DIR
from style import BLUE, GRAY, RED, WHITE, YELLOW, apply_global_style


def get_mode(series):
    """Calculate the mode of a series, ignoring empty strings and NaNs."""
    m = series.dropna()
    m = m[m.astype(str).str.strip() != ""]
    if m.empty:
        return "N/A"
    return m.mode().iloc[0]


def run():
    logger.info("Generating Graph 2: Multidimensional Whaler Profiles (Dynamic)")

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

    # Define locations and features
    locations = {
        "New Bedford": {"filter": lambda x: str(x) == "New Bedford", "color": BLUE},
        "Rochester": {"filter": lambda x: str(x) == "Rochester", "color": RED},
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
        "Должность": "rank",
    }

    # Calculate profiles dynamically
    profiles = []
    for name, config in locations.items():
        subset = df[df["origin"].apply(config["filter"])].copy()
        profile = {"name": name, "color": config["color"]}
        for label, col in features.items():
            profile[label] = get_mode(subset[col])
        profiles.append(profile)

    # Prepare axis information (types, ranges, categories)
    axes_info = {}
    for label, _col in features.items():
        vals = [p[label] for p in profiles if p[label] != "N/A"]
        if label == "Рост (см)":
            # Numerical axis with padding
            h_min = min(vals) if vals else 160
            h_max = max(vals) if vals else 180
            axes_info[label] = {"type": "num", "min": h_min - 2, "max": h_max + 2}
        else:
            # Categorical axis with sorted unique values
            unique_cats = sorted(set(vals))
            axes_info[label] = {"type": "cat", "cats": unique_cats}

    def normalize(val, info):
        """Map values to a 0-1 range for consistent plotting across multiple axes."""
        if val == "N/A":
            return 0.5
        if info["type"] == "num":
            return (val - info["min"]) / (info["max"] - info["min"])
        cats = info["cats"]
        if len(cats) <= 1:
            return 0.5
        return cats.index(val) / (len(cats) - 1)

    # Plotting
    apply_global_style()
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor(WHITE)

    n_axes = len(features)
    x_indices = np.arange(n_axes)
    feature_labels = list(features.keys())

    # Draw vertical axes
    for i in x_indices:
        ax.axvline(i, color=GRAY, linewidth=0.8, alpha=0.5, zorder=1)

    # Sort profiles to handle label stacking on the first axis
    sorted_profiles = sorted(
        profiles,
        key=lambda p: normalize(p[feature_labels[0]], axes_info[feature_labels[0]]),
    )
    last_y = -1.0

    for p in sorted_profiles:
        y_vals = [normalize(p[label], axes_info[label]) for label in feature_labels]

        # Stacking logic for labels
        y_label = y_vals[0]
        if abs(y_label - last_y) < 0.05:
            y_label += 0.04
        last_y = y_label

        ax.plot(
            x_indices,
            y_vals,
            color=p["color"],
            linewidth=5,
            alpha=0.8,
            zorder=3,
            marker="o",
            markersize=12,
            markeredgecolor="white",
            markeredgewidth=1.5,
        )

        ax.text(
            x_indices[0] - 0.1,
            y_label,
            p["name"],
            color=p["color"],
            ha="right",
            va="center",
            fontweight="bold",
            fontsize=14,
        )

    # Style axes
    ax.set_xticks(x_indices)
    ax.set_xticklabels(feature_labels, fontsize=14, fontweight="bold", color=GRAY)
    ax.set_yticks([])
    ax.set_ylim(-0.1, 1.1)

    # Add tick labels for each vertical axis
    for i, label in enumerate(feature_labels):
        info = axes_info[label]
        if info["type"] == "num":
            # Display 5 sample values on numerical axis
            ticks = np.linspace(info["min"], info["max"], 5)
            for val in ticks:
                pos = (val - info["min"]) / (info["max"] - info["min"])
                ax.text(
                    i + 0.08,
                    pos,
                    f"{val:.1f}",
                    color=GRAY,
                    fontsize=11,
                    va="center",
                    bbox={
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.7,
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
                    fontsize=11,
                    va="center",
                    bbox={
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.7,
                        "pad": 0.5,
                    },
                )

    for spine in ax.spines.values():
        spine.set_visible(False)

    # Dynamic Titles
    main_title = "Профили китобоев: моряки из New Bedford и Rochester схожи по типажу,\nв то время как экипажи с Azores выделяются иным цветом кожи и рангами"
    subtitle = "Самые распространённые физические характеристики (рост, цвет кожи, волос) и должности среди жителей трёх портовых городов"

    fig.suptitle(
        main_title,
        fontsize=20,
        fontweight="bold",
        x=0.5,
        y=0.96,
        ha="center",
        color=GRAY,
    )
    plt.text(
        0.5,
        0.89,
        subtitle,
        fontsize=14,
        ha="center",
        va="center",
        transform=fig.transFigure,
        color=GRAY,
    )

    footer_text = "Источник: Исторические данные об экипажах китобойных судов (XIX - начало XX века). Цвет волос использован вместо цвета глаз из-за отсутствия данных."
    plt.text(
        0.01,
        0.01,
        footer_text,
        fontsize=9,
        ha="left",
        transform=fig.transFigure,
        color=GRAY,
        style="italic",
    )

    plt.tight_layout(rect=(0.05, 0.05, 0.95, 0.86))

    output_path = OUTPUT_DIR / f"graph2.{FORMAT}"
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    logger.success(f"Graph 2 saved to {output_path}")


if __name__ == "__main__":
    run()
