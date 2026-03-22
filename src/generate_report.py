from pathlib import Path

import pandas as pd


def generate():
    raw_file = "data/raw/original_crew_dataset.csv"
    output_path = Path("output/research_report.md")

    print(f"Reading {raw_file}...")
    df = pd.read_csv(raw_file, engine="python", on_bad_lines="skip")

    # Очистка
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["height_feet"] = pd.to_numeric(df["height_feet"], errors="coerce")
    df["height_inches"] = pd.to_numeric(df["height_inches"], errors="coerce")
    df["height_cm"] = (df["height_feet"] * 30.48) + (df["height_inches"] * 2.54)
    df["origin"] = df["birthplace"].fillna(df["res_city"])

    clean_df = df.dropna(subset=["age", "height_cm", "origin"]).copy()
    clean_df = clean_df[(clean_df["age"] > 10) & (clean_df["age"] < 75)]
    clean_df = clean_df[(clean_df["height_cm"] > 140) & (clean_df["height_cm"] < 215)]

    # Выбираем топ-20 портов по объему данных
    top_origins = clean_df["origin"].value_counts().head(20).index
    filtered_df = clean_df[clean_df["origin"].isin(top_origins)]

    # 1. Анализ распределения ВОЗРАСТА
    age_stats = (
        filtered_df.groupby("origin")["age"]
        .agg(
            [
                "count",
                "mean",
                "median",
                lambda x: x.quantile(0.25),
                lambda x: x.quantile(0.75),
                lambda x: (x < 19).mean() * 100,  # Процент подростков
            ]
        )
        .reset_index()
    )

    age_stats.columns = [
        "Origin",
        "Count",
        "Mean Age",
        "Median Age",
        "25%",
        "75%",
        "% Teens (<19)",
    ]
    age_stats["IQR"] = age_stats["75%"] - age_stats["25%"]

    # 2. Анализ распределения РОСТА
    height_stats = (
        filtered_df.groupby("origin")["height_cm"]
        .agg(
            [
                "mean",
                "median",
                "std",
                lambda x: x.max() - x.min(),  # Размах
            ]
        )
        .reset_index()
    )
    height_stats.columns = [
        "Origin",
        "Mean Height",
        "Median Height",
        "Std Dev",
        "Range",
    ]

    # 3. Дополнительные ответы на вопросы
    # — сколько матросов было из ферм
    farm_mask = (
        df["birthplace"].str.contains("farm", case=False, na=False)
        | df["res_city"].str.contains("farm", case=False, na=False)
        | df["remarks"].str.contains("farm", case=False, na=False)
    )
    farm_count = df[farm_mask].shape[0]

    # — сколько темнокожих в общей сложности из всех матросов
    dark_skin_labels = ["dark", "black", "colored", "mulatto", "negro", "brown"]
    dark_skin_mask = df["skin"].str.lower().str.strip().isin(dark_skin_labels)
    dark_skin_count = df[dark_skin_mask].shape[0]

    # — какой средний возраст помощника капитана/самого капитанов
    officer_ranks = ["Master", "1st Mate", "2nd Mate", "3rd Mate", "4th Mate"]
    officers = df[df["rank"].isin(officer_ranks)].copy()
    officers["age"] = pd.to_numeric(officers["age"], errors="coerce")
    avg_officer_age = officers["age"].mean()

    # Формирование отчета
    report = []
    report.append("# Анализ распределений: Возраст и Рост китобоев\n")

    report.append("## 0. Ответы на ключевые вопросы")
    report.append(f"- **Сколько матросов было из ферм:** {farm_count}")
    report.append(
        f"- **Сколько темнокожих матросов в общей сложности:** {dark_skin_count}"
    )
    report.append(
        f"- **Средний возраст командного состава (капитаны и помощники):** {avg_officer_age:.1f} лет\n"
    )

    report.append(
        "> **Методология:** Мы отказались от простых средних в пользу анализа структуры популяций. "
        "Среднее значение может быть одинаковым, но за ним скрываются разные социальные модели.\n"
    )

    report.append("## 1. Возрастная структура портов")
    report.append(
        "Здесь мы ищем разницу между 'молодежными' городами и городами с широким возрастным разбросом.\n"
    )

    report.append("| Порт | Медиана | IQR (разброс) | % Подростков | Характер |")
    report.append("| :--- | :--- | :--- | :--- | :--- |")

    for _, row in age_stats.sort_values(by="% Teens (<19)", ascending=False).iterrows():
        # Определяем характер распределения
        if row["% Teens (<19)"] > 25:
            char = "Кузница кадров (много юнцов)"
        elif row["IQR"] > 10:
            char = "Разновозрастный (смешанный)"
        else:
            char = "Профессиональное ядро"

        report.append(
            f"| {row['Origin']} | {row['Median Age']:.1f} | {row['IQR']:.1f} | {row['% Teens (<19)']:.1f}% | {char} |"
        )

    report.append("\n## 2. Аномалии в распределении роста")
    report.append(
        "Высокое стандартное отклонение (Std Dev) указывает на генетическое или социальное разнообразие группы.\n"
    )

    report.append("| Порт | Средний рост | Std Dev | Range (см) |")
    report.append("| :--- | :--- | :--- | :--- |")
    for _, row in (
        height_stats.sort_values(by="Std Dev", ascending=False).head(10).iterrows()
    ):
        report.append(
            f"| {row['Origin']} | {row['Mean Height']:.1f} | {row['Std Dev']:.2f} | {row['Range']:.1f} |"
        )

    report.append("\n## 3. Портрет типичного китобоя по портам")
    report.append(
        "На основе модальных значений мы составили портрет наиболее часто встречающегося моряка для ключевых регионов.\n"
    )

    portrait_locations = {
        "New Bedford": lambda x: str(x) == "New Bedford",
        "Rochester": lambda x: str(x) == "Rochester",
        "Azores": lambda x: any(
            island.lower() in str(x).lower()
            for island in ["Azores", "Fayal", "Pico", "Flores", "Sao Jorge"]
        ),
    }

    report.append("| Регион | Возраст | Рост | Волосы | Глаза | Кожа | Ранг |")
    report.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")

    for loc_name, filter_fn in portrait_locations.items():
        subset = df[df["origin"].apply(filter_fn)].copy()

        def get_mode(series):
            m = series.dropna()
            m = m[m.astype(str).str.strip() != ""]
            m = m.mode()
            return m.iloc[0] if not m.empty else "N/A"

        age = (
            int(get_mode(subset["age"])) if get_mode(subset["age"]) != "N/A" else "N/A"
        )
        height = get_mode(subset["height_cm"])
        if height != "N/A":
            height = f"{height:.1f}"
        hair = get_mode(subset["hair"])
        eye = get_mode(subset["eye"])
        skin = get_mode(subset["skin"])
        rank = get_mode(subset["rank"])

        report.append(
            f"| {loc_name} | {age} | {height} см | {hair} | {eye} | {skin} | {rank} |"
        )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    print(f"Distribution report generated at {output_path}")


if __name__ == "__main__":
    generate()
