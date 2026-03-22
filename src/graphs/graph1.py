import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

from config import DPI, FORMAT, OUTPUT_DIR, ROOT_DIR
from style import BLUE, GRAY, LIGHT_GRAY, RED, WHITE, apply_global_style

# Constants for classification
WORLD_KEYWORDS = [
    "Azores",
    "Brava",
    "Cape Verde",
    "Fayal",
    "Germany",
    "Pico",
    "Saint Helena",
    "Flores",
    "Sao Jorge",
    "Southampton",
    "Portugal",
    "Great Britain",
    "England",
    "Ireland",
    "Scotland",
    "Western Islands",
    "Sandwich Islands",
    "Hawaii",
    "Sweden",
    "Swead",
    "Denmark",
    "Norway",
    "Prussia",
    "Holland",
    "France",
    "Spain",
    "Italy",
    "Canary Islands",
    "Madeira",
    "Africa",
    "Australia",
    "New Zealand",
    "China",
    "Japan",
    "Bermuda",
    "Cuba",
    "Jamaica",
    "West Indies",
    "Brazil",
    "Chile",
    "Peru",
]


def classify_location(city):
    if pd.isna(city) or city.lower() in ["unknown", "not stated", "-", "`", ""]:
        return None

    city_str = str(city)
    for kw in WORLD_KEYWORDS:
        if kw.lower() in city_str.lower():
            return "World"

    # Check for US state abbreviations
    us_states = [
        ", MA",
        ", CT",
        ", RI",
        ", NY",
        ", ME",
        ", NH",
        ", VT",
        ", PA",
        ", MD",
        ", VA",
        ", NC",
        ", SC",
        ", GA",
        ", FL",
    ]
    for state in us_states:
        if state in city_str.upper():
            return "USA"

    # Default to USA if it's one of the known major US ports
    us_ports = [
        "New Bedford",
        "New London",
        "New York",
        "Boston",
        "Fairhaven",
        "Dartmouth",
        "Stonington",
        "Nantucket",
        "Westport",
        "Groton",
        "Philadelphia",
        "Tiverton",
        "Rochester",
        "Salem",
        "Edgartown",
        "Tisbury",
        "Fair Haven",
        "Norwich",
        "New York City",
        "Wareham",
        "Chilmark",
        "Falmouth",
        "Baltimore",
        "Waterford",
        "Mattapoisett",
        "Marion",
        "Providence",
        "Hartford",
        "Fall River",
        "Albany",
        "Freetown",
        "Taunton",
        "Brooklyn",
        "Provincetown",
        "Mystic",
        "New Haven",
        "Sagamore",
        "Bridgeport",
        "Hyannis",
        "Osterville",
        "Barnstable",
    ]
    for port in us_ports:
        if port.lower() in city_str.lower():
            return "USA"

    return "USA"


def clean_city_name(city):
    if pd.isna(city):
        return city
    city_str = str(city)
    if "," in city_str:
        city_str = city_str.split(",")[0]
    return city_str.strip()


def plot_panel(
    ax, stats, title, color_base, highlight_color_young, highlight_color_old
):
    # Sort by age
    stats = stats.sort_values(by="mode_age", ascending=False)

    # Colors: Neutral with accents for min/max ages
    min_age = stats["mode_age"].min()
    max_age = stats["mode_age"].max()

    y_pos = np.arange(len(stats))
    height = 0.6

    # Set X-axis scale from 0 to 26 with step 2
    ax.set_xlim(0, 26)
    ax.set_xticks(range(0, 27, 2))

    for i, (_, row) in enumerate(stats.iterrows()):
        age = row["mode_age"]

        color = color_base
        if age == min_age:
            color = highlight_color_young
        elif age == max_age:
            color = highlight_color_old

        # 1. Drop Shadow Effect
        shadow_offset_x = 0.15
        shadow_offset_y = -0.05
        shadow_rect = patches.FancyBboxPatch(
            (shadow_offset_x, i - height / 2 + shadow_offset_y),
            age,
            height,
            boxstyle="round,pad=0,rounding_size=0.2",
            ec="none",
            fc="gray",
            alpha=0.2,
            zorder=1,
        )
        ax.add_patch(shadow_rect)

        # 2. Main Bar with Rounded Corners
        rect = patches.FancyBboxPatch(
            (0, i - height / 2),
            age,
            height,
            boxstyle="round,pad=0,rounding_size=0.2",
            ec="none",
            fc=color,
            zorder=2,
        )
        ax.add_patch(rect)

        # Add labels
        ax.text(
            age + 0.5,
            i,
            f"{int(age)}",
            va="center",
            fontsize=10,
            color=GRAY,
            fontweight="bold",
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(stats["city"], color=GRAY)

    # Add padding to top and bottom to prevent clipping
    ax.set_ylim(-1, len(stats))

    ax.set_title(title, fontsize=14, pad=20, color=GRAY, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(GRAY)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Самый частый возраст", fontsize=12, color=GRAY)
    ax.grid(axis="x", linestyle="--", alpha=0.2, zorder=0)


def run():
    logger.info("Generating Graph 1: Age Distribution Comparison (USA vs World)")

    # Load data
    data_path = ROOT_DIR / "data" / "raw" / "original_crew_dataset.csv"
    try:
        df = pd.read_csv(data_path, engine="python", on_bad_lines="skip")
    except Exception as e:
        logger.error(f"Failed to read dataset: {e}")
        return

    # Preprocessing
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df = df.dropna(subset=["age", "res_city"])

    df["clean_city"] = df["res_city"].apply(clean_city_name)
    df["location_type"] = df["res_city"].apply(classify_location)

    df = df.dropna(subset=["location_type"])

    # Filter for valid ages (10 to 80)
    df = df[(df["age"] >= 10) & (df["age"] <= 80)]

    # Calculate mode age for each city and city counts
    usa_df = df[df["location_type"] == "USA"]
    world_df = df[df["location_type"] == "World"]

    # Target cities requested by the user
    target_usa = [
        "New London",
        "New York",
        "Salem",
        "Nantucket",
        "Albany",
        "Philadelphia",
    ]
    target_world = ["Germany"]

    def get_top_cities(subset_df, targets, top_n=20):
        # Initial top list by count
        top_list = subset_df["clean_city"].value_counts().head(top_n).index.tolist()
        # Add requested cities if missing
        for target in targets:
            if (
                target not in top_list
                and not subset_df[subset_df["clean_city"] == target].empty
            ):
                top_list.append(target)
        return top_list

    # Top cities for each location type
    top_20_usa_cities = get_top_cities(usa_df, target_usa, 20)
    top_20_world_cities = get_top_cities(world_df, target_world, 20)

    def get_city_stats(subset_df, cities):
        stats = []
        for city in cities:
            city_data = subset_df[subset_df["clean_city"] == city]
            if not city_data.empty:
                mode_res = city_data["age"].mode()
                if not mode_res.empty:
                    mode_age = mode_res[0]
                    stats.append({"city": city, "mode_age": mode_age})
        return pd.DataFrame(stats).sort_values(by="mode_age")

    usa_stats = get_city_stats(usa_df, top_20_usa_cities)
    world_stats = get_city_stats(world_df, top_20_world_cities)

    # Plotting
    apply_global_style()
    # Narrower figure to reduce middle gap
    fig, (ax_world, ax_usa) = plt.subplots(1, 2, figsize=(14, 8))
    fig.patch.set_facecolor(WHITE)

    plot_panel(ax_world, world_stats, "регионы мира", LIGHT_GRAY, BLUE, RED)
    plot_panel(ax_usa, usa_stats, "города США", LIGHT_GRAY, BLUE, RED)

    # Titles
    main_title = "Китобои-американцы были моложе нанятых из других стран"
    subtitle = "Самый распространённый возраст среди участников экспедиций из портов США\n(выбраны ТОП-20 мест по числу выходцев)"

    fig.suptitle(
        main_title,
        fontsize=20,
        fontweight="bold",
        x=0.5,
        y=0.96,
        ha="center",
        va="center",
        color=GRAY,
    )
    plt.text(
        0.5,
        0.87,
        subtitle,
        fontsize=14,
        ha="center",
        va="center",
        transform=fig.transFigure,
        color=GRAY,
    )

    # Footer
    footer_text = "Источник: New Bedford Whaling Museum, Nantucket Historical Association and Mystic Seaport Museum «American Offshore Whaling : About Crew Lists»"
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

    plt.tight_layout(rect=(0, 0.03, 1, 0.84))
    # Explicitly adjust space between subplots
    plt.subplots_adjust(wspace=0.25)

    # Save
    output_path = OUTPUT_DIR / f"graph1.{FORMAT}"
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()

    logger.success(f"Graph 1 saved to {output_path}")


if __name__ == "__main__":
    run()
