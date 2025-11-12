import csv
import math
import os
import random
from datetime import datetime, timedelta


def parse_float(value: str) -> float | None:
    if value is None:
        return None
    text = value.strip()
    if text == "" or text.upper() == "#N/A":
        return None
    # Remove thousands separators
    text = text.replace(",", "")
    try:
        return float(text)
    except ValueError:
        return None


def format_date(dt: datetime) -> str:
    # Match the CSV style like 7/21/2023 (no leading zeros)
    return f"{dt.month}/{dt.day}/{dt.year}"


def read_existing_daily(csv_path: str) -> list[tuple[datetime, float]]:
    results: list[tuple[datetime, float]] = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # Expect at least columns: Date, INR (others can exist and will be ignored)
        for row in reader:
            date_text = (row.get("Date") or "").strip()
            inr_value = parse_float(row.get("INR"))
            if not date_text:
                continue
            try:
                dt = datetime.strptime(date_text, "%m/%d/%Y")
            except ValueError:
                # Try alternative common patterns if needed
                parsed = None
                for pattern in ("%m/%d/%y", "%Y-%m-%d"):
                    try:
                        parsed = datetime.strptime(date_text, pattern)
                        break
                    except ValueError:
                        continue
                if parsed is None:
                    continue
                dt = parsed

            if inr_value is None:
                # Skip rows where INR is not available
                continue
            results.append((dt, inr_value))
    return results


def generate_random_walk(start_date: datetime, start_price: float, end_date: datetime) -> list[tuple[datetime, float]]:
    if start_date.date() >= end_date.date():
        return []
    generated: list[tuple[datetime, float]] = []
    current_date = start_date
    current_price = float(start_price)
    while current_date.date() < end_date.date():
        current_date = current_date + timedelta(days=1)
        change = random.uniform(-0.005, 0.005)  # ±0.5%
        current_price = current_price * (1.0 + change)
        generated.append((current_date, current_price))
    return generated


def write_updated(output_path: str, rows: list[tuple[datetime, float]]):
    # Deduplicate by date and ensure ascending order
    by_date: dict[str, float] = {}
    for dt, price in rows:
        key = dt.date().isoformat()
        by_date[key] = price  # latest wins, though there shouldn't be duplicates

    # Sort by actual date
    sorted_items = sorted(((datetime.fromisoformat(k), v) for k, v in by_date.items()), key=lambda x: x[0])

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Date", "INR"])
        for dt, price in sorted_items:
            writer.writerow([format_date(dt), f"{price:.2f}"])


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_csv = os.path.join(base_dir, "Daily.csv")
    output_csv = os.path.join(base_dir, "Updated_Daily.csv")

    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input file not found: {input_csv}")

    existing = read_existing_daily(input_csv)
    if not existing:
        raise ValueError("No valid (Date, INR) rows found in Daily.csv")

    # Ensure existing are sorted by date
    existing.sort(key=lambda x: x[0])

    last_date, last_price = existing[-1]
    today = datetime.today()

    generated = generate_random_walk(last_date, last_price, today)

    all_rows = existing + generated
    write_updated(output_csv, all_rows)


if __name__ == "__main__":
    main()


