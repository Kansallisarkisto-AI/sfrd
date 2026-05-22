import json
import argparse
from pathlib import Path
import polars as pl
import xlsxwriter
from xlsxwriter.utility import xl_col_to_name
import re
import pickle

def interpolate_color(c1, c2, t):
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))

def rgb_to_hex(rgb):
    return "#{:02X}{:02X}{:02X}".format(*rgb)

def make_gradient(n):
    red = (248, 105, 107)    # Excel-ish red
    yellow = (255, 235, 132) # Excel-ish yellow
    green = (99, 190, 123)   # Excel-ish green

    colors = []
    for i in range(n):
        t = i / (n - 1) if n > 1 else 0

        if t < 0.5:
            # red → yellow
            c = interpolate_color(red, yellow, t * 2)
        else:
            # yellow → green
            c = interpolate_color(yellow, green, (t - 0.5) * 2)

        colors.append(rgb_to_hex(c))

    return colors

def get_thresholds(series, n):
    return [float(series.drop_nulls().drop_nans().quantile(i / n)) for i in range(n + 1)]

def get_thresholds_linear(n, start=0.85):
    return [start + (1.0 - start) * (i / n) for i in range(n + 1)]

parser = argparse.ArgumentParser(description="Convert sfrd json to xlsx")

parser.add_argument(
    "json_path",
    type=str,
    default='test.json',
    help="Path to the json file"
)

parser.add_argument(
    "output_path",
    type=str,
    default='out.xlsx',
    help="Path to the output xlsx file"
)

args = parser.parse_args()

with open(args.json_path, "r") as f:
    output_dicts = json.load(f)

output_key = Path(args.json_path).stem

df = pl.from_dicts(output_dicts, infer_schema_length=10000)

ordered_cols = list(output_dicts[0].keys())

extra_cols = [
    c for c in df.columns
    if c not in ordered_cols
]

all_cols = ordered_cols + extra_cols

with xlsxwriter.Workbook(
    args.output_path,
    {"nan_inf_to_errors": True, "strings_to_urls": False, "strings_to_formulas": False}
) as workbook:

    df2 = df.select([
        x for x in all_cols
        if not x.startswith("CONF_")
    ])

    df2.write_excel(
        workbook=workbook,
        worksheet="Simple",
        autofit=True
    )

    df = df.select(all_cols)

    df.write_excel(
        workbook=workbook,
        worksheet="Complex",
        autofit=True
    )

    complex_worksheet = workbook.get_worksheet_by_name("Complex")

    for i in range(0, len(all_cols)):
        if all_cols[i].startswith("CONF_"):
            complex_worksheet.set_column(
                i,
                i,
                None,
                None,
                {"hidden": True, "level": 1}
            )

    n_bands = 5
    colors = make_gradient(n_bands)

    n_rows = len(df)

    for i, col in enumerate(all_cols):

        if col.startswith("CONF_") and not col.startswith("CONF_||") and not col.startswith("CONF___"):

            target_col = i - 1

            conf_letter = xl_col_to_name(i)
            target_letter = xl_col_to_name(target_col)

            try:
                thresholds = get_thresholds_linear(n_bands)
            except:
                continue

            cell_range = (
                f"{target_letter}2:"
                f"{target_letter}{n_rows+1}"
            )

            for j in range(n_bands):
                low = f"{thresholds[j]:.6f}"
                high = f"{thresholds[j + 1]:.6f}"

                fmt = workbook.add_format({
                    "bg_color": colors[j]
                })

                if j == 0:
                    formula = (
                        f"=${conf_letter}2<{high}"
                    )

                elif j == n_bands - 1:
                    formula = (
                        f"=${conf_letter}2>={low}"
                    )

                else:
                    formula = (
                        f"=AND("
                        f"${conf_letter}2>={low}, "
                        f"${conf_letter}2<{high}"
                        f")"
                    )

                complex_worksheet.conditional_format(
                    cell_range,
                    {
                        "type": "formula",
                        "criteria": formula,
                        "format": fmt,
                    }
                )
