import warnings

import pandas as pd

# Подавляем предупреждения о плохих строках для чистоты вывода
warnings.filterwarnings("ignore")


def get_mode(series):
    # Очищаем от пустых строк и N/A перед вычислением моды
    clean = series.dropna().astype(str).str.strip()
    clean = clean[clean != ""]
    return clean.mode().iloc[0] if not clean.mode().empty else "N/A"


def format_height(f, i):
    if pd.isna(f) or pd.isna(i):
        return "N/A"
    return f"{int(f)}'{int(i)}\""


try:
    # Загрузка с пропуском проблемных строк и использованием движка python
    df = pd.read_csv(
        "data/raw/original_crew_dataset.csv", engine="python", on_bad_lines="skip"
    )

    # Приведение типов
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["height_feet"] = pd.to_numeric(df["height_feet"], errors="coerce")
    df["height_inches"] = pd.to_numeric(df["height_inches"], errors="coerce")

    locations = {
        "Нью Бэдфорд": df[
            df["res_city"].str.contains("New Bedford", na=False, case=False)
        ],
        "Рочестер": df[df["res_city"].str.contains("Rochester", na=False, case=False)],
        "Азорские острова": df[
            df["res_country"].str.contains("Portugal", na=False, case=False)
            | df["birthplace"].str.contains("Azores", na=False, case=False)
        ],
    }

    for name, subset in locations.items():
        if subset.empty:
            print(f"{name}: данные не найдены")
            continue

        # Возраст
        age_series = subset["age"].dropna()
        age = int(age_series.mode().iloc[0]) if not age_series.mode().empty else "N/A"

        # Рост (самая частая пара футы-дюймы)
        h_subset = subset[["height_feet", "height_inches"]].dropna()
        if not h_subset.empty:
            h_mode = h_subset.value_counts().idxmax()
            height = format_height(h_mode[0], h_mode[1])
        else:
            height = "N/A"

        hair = get_mode(subset["hair"])
        eye = get_mode(subset["eye"])
        skin = get_mode(subset["skin"])
        rank = get_mode(subset["rank"])

        print(
            f"{name}: самый распространённый возраст {age} - рост {height} - цвет волос {hair} - цвет глаз {eye} - цвет кожи {skin} - ранг {rank}"
        )

except Exception as e:
    print(f"Ошибка при обработке: {e}")
