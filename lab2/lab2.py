import os
import csv
from collections import defaultdict
from itertools import product
import pandas as pd


def read_probabilities(file_path):
    df = pd.read_csv(file_path, header=None, sep=";", usecols=[0, 1])
    df.columns = ["key", "prob"]
    df["key"] = df["key"].str.strip().str.replace(";", "", regex=False)
    return dict(zip(df["key"], df["prob"]))


def read_pair_values(file_path):
    df = pd.read_csv(file_path, header=0, sep=";")
    df.columns = [col.strip().replace(";", "") for col in df.columns]
    df.index = df.iloc[:, 0].str.strip().replace(";", "", regex=False)
    df = df.iloc[:, 1:]
    df.index.name = None
    return df


def generate_combinations(groups, group_keys):
    group_values = [[f"{g}.{v}." for v in groups[g]] for g in group_keys]
    return list(product(*group_values))


def compute_pair_product(combo, pair_table):
    result = 1.0
    for i, a in enumerate(combo):
        for b in combo[i + 1 :]:
            val = pair_table.at[b, a]
            result *= float(val) + 1
    return result


def read_table3(file_path):
    df = pd.read_csv(file_path, header=0, sep=";")
    df.columns = [col.strip().replace(";", "") for col in df.columns]
    df.index = df.iloc[:, 0].str.strip().replace(";", "", regex=False)
    df = df.iloc[:, 1:]
    df.index.name = None
    return df


def compute_table3_scores(combos, table3):
    results = []
    alt_cols = list(table3.columns)
    for combo in combos:
        row = list(combo)
        for alt in alt_cols:
            try:
                product = 1.0
                for alt_key in combo:
                    value = float(table3.at[alt_key, alt])
                    product *= value + 1
                row.append(product)
            except Exception as e:
                print(f"Error processing combo {combo} with {alt}: {e}")
                row.append(None)
        results.append(row)
    return results, alt_cols


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    table1_path = os.path.join(base_dir, "table1.csv")
    table2_path = os.path.join(base_dir, "table2.csv")
    output_path = os.path.join(base_dir, "combinations_with_p.csv")
    alt_output_path = os.path.join(base_dir, "alternative_probabilities.csv")

    # Define groups
    groups = {
        1: [1, 2, 3],
        2: [1, 2, 3, 4, 5],
        3: [1, 2, 3, 4],
        4: [1, 2, 3, 4],
        5: [1, 2, 3, 4],
        6: [1, 2],
        7: [1, 2, 3],
    }
    group_keys = sorted(groups.keys())

    # Read data
    prob_dict = read_probabilities(table1_path)
    pair_table = read_pair_values(table2_path)
    combinations = generate_combinations(groups, group_keys)

    # Calculate probabilities
    results = []
    for combo in combinations:
        try:
            P = 1.0
            for key in combo:
                if key not in prob_dict:
                    raise KeyError(f"Missing probability for {key}")
                P *= float(prob_dict[key])
            C = compute_pair_product(combo, pair_table)
            PC = P * C
            results.append(list(combo) + [P, C, PC])
        except Exception as e:
            print(f"Skipping combination {combo} due to error: {e}")
            continue

    # Normalize PC values
    PC_values = [row[-1] for row in results]
    PC_total = sum(PC_values)
    PC_norm = [pc / PC_total if PC_total != 0 else 0 for pc in PC_values]

    # Append normalized results
    for i, norm in enumerate(PC_norm):
        results[i].append(norm)

    # Write combinations and scores to CSV
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        headers = [f"Group{i}" for i in range(1, 8)] + ["P", "C", "PC", "PC-Norm"]
        writer.writerow(headers)
        writer.writerows(results)

    # Compute alternative probabilities
    alt_probs = defaultdict(float)
    for row in results:
        norm_result = row[-1]
        for alt in row[:7]:
            alt_probs[alt] += norm_result

    with open(alt_output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Alternative", "Probability"])
        for alt, prob in sorted(alt_probs.items()):
            writer.writerow([alt, prob])

    print(f"Saved {len(results)} combinations to {output_path}")
    print(f"Saved alternative probabilities to {alt_output_path}")

    table3_path = os.path.join(base_dir, "table3.csv")
    table3 = read_table3(table3_path)

    # Prepare combinations from previous output
    df_combinations = pd.read_csv(output_path)
    combo_cols = [f"Group{i}" for i in range(1, 8)]
    combos_only = df_combinations[combo_cols].values.tolist()
    last_col = df_combinations["PC-Norm"].tolist()

    # Compute new values for 8.x alternatives
    scored_combos, alt_columns = compute_table3_scores(combos_only, table3)

    # Append PC-Norm column
    for i, row in enumerate(scored_combos):
        row.append(last_col[i])

    # Write extended table to file
    new_output_path = os.path.join(base_dir, "extended_combinations.csv")
    with open(new_output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # Rearranged headers: PC-Norm immediately after group columns
        headers = combo_cols + ["PC-Norm"] + alt_columns

        # Move PC-Norm column to correct position in each row
        scored_combos = [row[:7] + [row[-1]] + row[7:-1] for row in scored_combos]

        writer.writerow(headers)
        writer.writerows(scored_combos)

    print(f"Saved extended combinations with 8.x scores to {new_output_path}")

    # Шлях до extended_combinations.csv
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Шлях до extended_combinations.csv
    extended_path = os.path.join(base_dir, "extended_combinations.csv")
    df = pd.read_csv(extended_path)

    # Знаходження колонок 8.x.
    score_columns = [col for col in df.columns if col.startswith("8.")]
    score_df = df[score_columns]

    # Обчислення суми по кожному рядку
    row_sums = score_df.sum(axis=1)

    # Нормалізація і додавання нових колонок
    for col in score_columns:
        norm_col = f"{col}-Norm"
        df[norm_col] = df[col] / row_sums

    # Перезаписати існуючий файл з новими стовпцями
    df.to_csv(extended_path, index=False)

    print(f"Updated extended_combinations.csv with normalized 8.x columns")

    # Нормалізація і додавання нових колонок
    for col in score_columns:
        norm_col = f"{col}-Norm"
        df[norm_col] = df[col] / row_sums

    # Додавання нових колонок з множенням на PC-Norm
    for col in score_columns:
        norm_col = f"{col}-Norm"
        weighted_col = f"{col}-Weighted"
        df[weighted_col] = df[norm_col] * df["PC-Norm"]

    # Перезаписати існуючий файл з новими стовпцями
    df.to_csv(extended_path, index=False)

    print(f"Updated extended_combinations.csv with normalized and weighted 8.x columns")

    # Знаходження колонок 8.x-Weighted.
    weighted_columns = [col for col in df.columns if col.endswith("-Weighted")]
    weighted_df = df[weighted_columns]

    # Обчислення суми по кожному стовпцю
    column_sums = weighted_df.sum(axis=0)

    # Нормалізація суми по кожному стовпцю для отримання імовірностей
    total_sum = column_sums.sum()
    normalized_sums = column_sums / total_sum if total_sum != 0 else column_sums

    # Створення нової таблиці для збереження імовірностей
    probabilities_output_path = os.path.join(
        base_dir, "alternative_probabilities_8x_weighted.csv"
    )
    with open(probabilities_output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Alternative", "Probability"])

        # Записати кожну альтернативу та її ймовірність
        for alt, prob in normalized_sums.items():
            writer.writerow([alt, prob])

    print(
        f"Saved alternative probabilities for 8.x-weighted columns to {probabilities_output_path}"
    )
