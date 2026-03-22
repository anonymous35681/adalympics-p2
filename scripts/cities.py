import csv
from collections import Counter


def normalize_country(country):
    if not country:
        return "Unknown"
    c = country.strip().lower()
    if c in ["usa", "united states", "u.s.a.", "u.s.", "america"]:
        return "USA"
    if "cape verde" in c or "brava" in c:
        return "Cape Verde"
    if "azores" in c or "pico" in c or "fayal" in c or "st michael" in c:
        return "Azores (Portugal)"
    if "england" in c or "great britain" in c or "uk" in c:
        return "Great Britain"
    if "germany" in c or "prussia" in c:
        return "Germany"
    return country.strip().title()


def process_cities():
    input_file = "data/raw/original_crew_dataset.csv"

    city_country_counts = Counter()

    try:
        with open(input_file, encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
            except Exception:
                return

            col_map = {name.strip('"'): i for i, name in enumerate(header)}

            res_city_idx = col_map.get("res_city")
            res_country_idx = col_map.get("res_country")
            birthplace_idx = col_map.get("birthplace")
            citizenship_idx = col_map.get("citizenship")

            for row in reader:
                try:
                    city = ""
                    country = ""

                    # Извлекаем город и страну (приоритет - birthplace)
                    if birthplace_idx is not None and len(row) > birthplace_idx:
                        bp = row[birthplace_idx]
                        if bp and "," in bp:
                            parts = [p.strip() for p in bp.split(",")]
                            city = parts[0]
                            country = parts[-1]
                            if len(parts) == 2 and len(parts[1]) == 2:
                                country = "USA"

                    if (
                        not city
                        and res_city_idx is not None
                        and len(row) > res_city_idx
                    ):
                        city = row[res_city_idx]
                    if (
                        not country
                        and res_country_idx is not None
                        and len(row) > res_country_idx
                    ):
                        country = row[res_country_idx]
                    if (
                        not country
                        and citizenship_idx is not None
                        and len(row) > citizenship_idx
                    ):
                        country = row[citizenship_idx]

                    if city or country:
                        norm_country = normalize_country(country)
                        norm_city = city.strip().title() if city else "Unknown"
                        city_country_counts[(norm_city, norm_country)] += 1
                except Exception:
                    continue

    except Exception as e:
        print(f"Error: {e}")

    # Вывод в порядке убывания
    print(f"{'ГОРОД':<25} | {'СТРАНА':<20} | {'ЛЮДЕЙ'}")
    print("-" * 60)

    sorted_stats = city_country_counts.most_common(50)  # Топ-50 для наглядности
    for (city, country), count in sorted_stats:
        if city == "Unknown" and country == "Unknown":
            continue
        print(f"{city[:25]:<25} | {country[:20]:<20} | {count}")


if __name__ == "__main__":
    process_cities()
