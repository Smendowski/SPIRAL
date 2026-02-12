import re
import sys

MY_METHODS_KEYWORDS = ["CNN-AE", "ResNet18-AE", "PVT-v2-b1-AE"]
SPIRAL_KEYWORD = "SPIRAL"


def preference_score(name):
    if SPIRAL_KEYWORD in name:
        return 3
    if any(k in name for k in MY_METHODS_KEYWORDS):
        return 2
    return 1


input_file = "table.tex"
output_file = "table_formatted.tex"

with open(input_file, "r") as f:
    lines = f.readlines()

data_rows = []

for i, line in enumerate(lines):
    stripped = line.strip()
    if not stripped or stripped.startswith("%"):
        continue
    if (
        stripped.startswith("\\midrule")
        or stripped.startswith("\\toprule")
        or stripped.startswith("\\bottomrule")
        or stripped.startswith("\\hline")
    ):
        continue
    if "&" not in stripped:
        continue

    clean = re.sub(r"\\\\.*$", "", stripped).strip()
    parts = [p.strip() for p in clean.split("&")]

    if len(parts) < 2:
        continue

    name = parts[0]
    try:
        values = [float(v) for v in parts[1:]]
        data_rows.append((i, name, values))
    except ValueError:
        continue

if not data_rows:
    print("No data rows found!")
    sys.exit(1)

n_cols = len(data_rows[0][2])
print(f"Found {len(data_rows)} rows, {n_cols} columns")

best_per_col = {}
second_per_col = {}

for col_idx in range(n_cols):
    col_values = {name: vals[col_idx] for (_, name, vals) in data_rows}
    max_val = max(col_values.values())

    # Find highest value
    tied_best = [n for n, v in col_values.items() if v == max_val]
    best = sorted(tied_best, key=preference_score, reverse=True)[0]

    # Find second highest value
    remaining = {n: v for n, v in col_values.items() if n != best}
    second_val = max(remaining.values())
    tied_second = [n for n, v in remaining.items() if v == second_val]
    second = sorted(tied_second, key=preference_score, reverse=True)[0]

    best_per_col[col_idx] = best
    second_per_col[col_idx] = second


def format_value(val, name, col_idx):
    formatted = f"{val:.2f}"
    if name == best_per_col[col_idx]:
        return f"\\textbf{{{formatted}}}"
    elif name == second_per_col[col_idx]:
        return f"\\underline{{{formatted}}}"
    return formatted


row_map = {i: (name, vals) for (i, name, vals) in data_rows}

output_lines = []
for i, line in enumerate(lines):
    if i in row_map:
        name, vals = row_map[i]

        trail_match = re.search(r"(\\\\.*?)$", line.rstrip())
        trail = trail_match.group(1) if trail_match else " \\\\"

        formatted_vals = [
            format_value(v, name, col_idx) for col_idx, v in enumerate(vals)
        ]
        new_line = f"{name} & " + " & ".join(formatted_vals) + f" {trail}\n"
        output_lines.append(new_line)
    else:
        output_lines.append(line)

with open(output_file, "w") as f:
    f.writelines(output_lines)
