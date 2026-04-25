#!/usr/bin/env python3
import csv
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data" / "week2"

# Inputs (existing files)
BASELINE_IN = DATA_DIR / "baseline_data.csv"
ADVERSARIAL_IN = DATA_DIR / "adversarial_data.csv"

# Outputs (new files)
OUT_DIR = DATA_DIR
BASELINE_OUT = OUT_DIR / "baseline_labeled.csv"
ADVERSARIAL_OUT = OUT_DIR / "adversarial_labeled.csv"
COMBINED_OUT = OUT_DIR / "week2_labeled_all.csv"

LABEL_COL = "label"

def label_file(input_csv: Path, output_csv: Path, label_value: int) -> int:
    with input_csv.open("r", newline="", encoding="utf-8") as fin:
        reader = csv.DictReader(fin)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {input_csv}")

        fieldnames = [c for c in reader.fieldnames if c != LABEL_COL] + [LABEL_COL]

        rows_written = 0
        with output_csv.open("w", newline="", encoding="utf-8") as fout:
            writer = csv.DictWriter(fout, fieldnames=fieldnames)
            writer.writeheader()

            for row in reader:
                row[LABEL_COL] = str(label_value)
                writer.writerow({k: row.get(k, "") for k in fieldnames})
                rows_written += 1

    return rows_written


def append_file(input_csv: Path, output_csv: Path, write_header: bool) -> int:
    with input_csv.open("r", newline="", encoding="utf-8") as fin:
        reader = csv.DictReader(fin)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {input_csv}")

        rows_written = 0
        mode = "w" if write_header else "a"
        with output_csv.open(mode, newline="", encoding="utf-8") as fout:
            writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
            if write_header:
                writer.writeheader()

            for row in reader:
                writer.writerow(row)
                rows_written += 1

    return rows_written


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    n_base = label_file(BASELINE_IN, BASELINE_OUT, 0)
    n_adv = label_file(ADVERSARIAL_IN, ADVERSARIAL_OUT, 1)

    # Build one combined labeled file
    n_comb_base = append_file(BASELINE_OUT, COMBINED_OUT, write_header=True)
    n_comb_adv = append_file(ADVERSARIAL_OUT, COMBINED_OUT, write_header=False)

    print("Done.")
    print(f"Baseline labeled rows: {n_base}")
    print(f"Adversarial labeled rows: {n_adv}")
    print(f"Combined rows: {n_comb_base + n_comb_adv}")
    print(f"Outputs written under: {OUT_DIR}")


if __name__ == "__main__":
    main()