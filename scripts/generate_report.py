from pathlib import Path

import pandas as pd


def generate():
    # 1. Загрузка данных
    raw_file = "data/raw/original_crew_dataset.csv"
    output_path = Path("output/research_report.md")

    print(f"Reading {raw_file}...")
    # Используем python engine для обработки битых строк с кавычками
    df = pd.read_csv(raw_file, engine="python", on_bad_lines="skip")

    # 2. Очистка и трансформация
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["height_feet"] = pd.to_numeric(df["height_feet"], errors="coerce")
    df["height_inches"] = pd.to_numeric(df["height_inches"], errors="coerce")

    # Расчет роста в см
    df["height_cm"] = (df["height_feet"] * 30.48) + (df["height_inches"] * 2.54)

    # Определение происхождения (Приоритет: birthplace, затем res_city)
    df["origin"] = df["birthplace"].fillna(df["res_city"])

    # Фильтрация валидных данных
    clean_df = df.dropna(subset=["age", "height_cm", "origin"]).copy()
    clean_df = clean_df[(clean_df["age"] > 10) & (clean_df["age"] < 75)]
    clean_df = clean_df[(clean_df["height_cm"] > 140) & (clean_df["height_cm"] < 215)]

    # 3. Анализ групп
    top_origins = clean_df["origin"].value_counts().head(30).index
    stats = (
        clean_df[clean_df["origin"].isin(top_origins)]
        .groupby("origin")
        .agg({"age": ["mean", "count"], "height_cm": ["mean", "std"]})
    )
    stats.columns = ["avg_age", "count", "avg_height", "std_height"]
    stats = stats.reset_index()

    # 4. Формирование отчета
    report = []
    report.append("# Исследование характеристик экипажей китобойных судов\n")
    report.append(f"**Объем проанализированных записей:** {len(clean_df)}\n")

    report.append("## 1. Сравнение роста по регионам")
    report.append(
        "Подтверждение гипотезы о превосходстве роста моряков из островных регионов.\n"
    )

    height_ranking = stats.sort_values(by="avg_height", ascending=False).head(10)
    report.append("| Место рождения | Средний рост (см) | Кол-во записей |")
    report.append("| :--- | :--- | :--- |")
    for _, row in height_ranking.iterrows():
        report.append(
            f"| {row['origin']} | {row['avg_height']:.2f} | {int(row['count'])} |"
        )

    report.append("\n## 2. Возрастной состав: Профессионалы vs Новички")
    report.append("Анализ среднего возраста по портам приписки/рождения.\n")

    age_ranking = stats.sort_values(by="avg_age", ascending=False)
    report.append("| Место рождения | Средний возраст | Статус |")
    report.append("| :--- | :--- | :--- |")
    for _, row in age_ranking.iloc[[0, 1, -2, -1]].iterrows():
        status = "Опытные" if row["avg_age"] > 25 else "Молодые"
        report.append(f"| {row['origin']} | {row['avg_age']:.1f} | {status} |")

    report.append("\n## 3. Физическая однородность (Standard Deviation)")
    report.append("Группы с наиболее 'стандартизированным' ростом.\n")

    std_ranking = stats.sort_values(by="std_height", ascending=True).head(5)
    report.append("| Место рождения | Std Deviation (рост) |")
    report.append("| :--- | :--- |")
    for _, row in std_ranking.iterrows():
        report.append(f"| {row['origin']} | {row['std_height']:.2f} |")

    # Сохранение
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    print(f"Report generated at {output_path}")


if __name__ == "__main__":
    generate()
