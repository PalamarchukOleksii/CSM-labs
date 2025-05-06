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


def compute_scores(combos, table):
    results = []
    alt_cols = list(table.columns)
    for combo in combos:
        row = list(combo)
        for alt in alt_cols:
            try:
                product_res = 1.0
                for alt_key in combo:
                    value = float(table.at[alt_key, alt])
                    product_res *= value + 1
                row.append(product_res)
            except (KeyError, ValueError, TypeError) as e:
                print(f"Error processing combo {combo} with {alt}: {e}")
                row.append(None)
        results.append(row)
    return results, alt_cols


def calculate_pc_combinations(groups, prob_dict, pair_table):
    group_keys = sorted(groups.keys())
    combinations = generate_combinations(groups, group_keys)

    results = []
    for combo in combinations:
        try:
            p = 1.0
            for key in combo:
                if key not in prob_dict:
                    raise KeyError(f"Missing probability for {key}")
                p *= float(prob_dict[key])

            c = compute_pair_product(combo, pair_table)

            pc = p * c
            results.append(list(combo) + [p, c, pc])
        except (KeyError, ValueError, TypeError) as e:
            print(f"Skipping combination {combo} due to error: {e}")
            continue

    pc_values = [row[-1] for row in results]
    pc_total = sum(pc_values)
    pc_norm = [pc / pc_total if pc_total != 0 else 0 for pc in pc_values]

    for i, norm in enumerate(pc_norm):
        results[i].append(norm)

    return results


def save_combinations_with_p(results, output_path):
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        headers = [f"Group{i}" for i in range(1, 8)] + ["P", "C", "PC", "PC-Norm"]
        writer.writerow(headers)
        writer.writerows(results)
    print(f"Saved {len(results)} combinations to {output_path}")


def calculate_alternative_probabilities(results, output_path):
    alt_probs = defaultdict(float)
    for row in results:
        norm_result = row[-1]
        for alt in row[:7]:
            alt_probs[alt] += norm_result

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Alternative", "Probability"])
        for alt, prob in sorted(alt_probs.items()):
            writer.writerow([alt, prob])
    print(f"Saved alternative probabilities to {output_path}")


def process_data(df_combinations, table_data, output_path):
    combo_cols = [f"Group{i}" for i in range(1, 8)]
    combos_only = df_combinations[combo_cols].values.tolist()
    pc_norm_values = df_combinations["PC-Norm"].tolist()

    scored_combos, alt_columns = compute_scores(combos_only, table_data)

    result_df = pd.DataFrame()

    for i, col_name in enumerate(combo_cols):
        result_df[col_name] = [combo[i] for combo in combos_only]

    result_df["PC-Norm"] = pc_norm_values

    for i, alt_col in enumerate(alt_columns):
        result_df[alt_col] = [row[7 + i] for row in scored_combos]

    result_df.to_csv(output_path, index=False)

    prefix = alt_columns[0].split(".")[0] if alt_columns else ""
    print(f"Saved extended combinations with {prefix}.x scores to {output_path}")

    return output_path


def normalize_and_weight_columns(extended_path, probabilities_output_path, prefix="8."):
    df = pd.read_csv(extended_path)

    score_columns = [col for col in df.columns if col.startswith(prefix)]
    score_df = df[score_columns]

    row_sums = score_df.sum(axis=1)

    for col in score_columns:
        norm_col = f"{col}-Norm"
        df[norm_col] = df[col] / row_sums

        weighted_col = f"{col}-Weighted"
        df[weighted_col] = df[norm_col] * df["PC-Norm"]

    all_columns = list(df.columns)

    group_cols = [col for col in all_columns if col.startswith("Group")]
    pc_norm_col = ["PC-Norm"]
    prefix_cols = [
        col
        for col in all_columns
        if col.startswith(prefix)
        and not col.endswith("-Norm")
        and not col.endswith("-Weighted")
    ]
    norm_cols = [
        col for col in all_columns if col.endswith("-Norm") and col != "PC-Norm"
    ]
    weighted_cols = [col for col in all_columns if col.endswith("-Weighted")]

    new_order = group_cols + pc_norm_col + prefix_cols + norm_cols + weighted_cols

    df = df[new_order]

    df.to_csv(extended_path, index=False)
    print(f"Updated {extended_path} with normalized and weighted {prefix}x columns")

    weighted_columns = [
        col
        for col in df.columns
        if col.startswith(prefix) and col.endswith("-Weighted")
    ]
    weighted_df = df[weighted_columns]

    column_sums = weighted_df.sum(axis=0)

    total_sum = column_sums.sum()
    normalized_sums = column_sums / total_sum if total_sum != 0 else column_sums

    with open(probabilities_output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Alternative", "Probability"])
        for alt, prob in normalized_sums.items():
            alt_name = alt.split("-")[0]
            writer.writerow([alt_name, prob])

    print(
        f"Saved alt probabilities for {prefix}x-weighted columns to {probabilities_output_path}"
    )


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = "data"
    starting_alternatives_path = os.path.join(
        base_dir, DATA_DIR, "starting_alternatives.csv"
    )
    interrelationship_matrix_path = os.path.join(
        base_dir, DATA_DIR, "interrelationship_matrix.csv"
    )
    connection_matrix_8_path = os.path.join(
        base_dir, DATA_DIR, "connection_matrix_8.csv"
    )
    connection_matrix_9_path = os.path.join(
        base_dir, DATA_DIR, "connection_matrix_9.csv"
    )

    OUTPUT_DIR = "result"
    output_dir_path = os.path.join(base_dir, OUTPUT_DIR)
    os.makedirs(output_dir_path, exist_ok=True)
    comb_pc_path = os.path.join(output_dir_path, "combinations_pc.csv")
    alt_prob_path = os.path.join(
        output_dir_path, "alternative_probabilities_1_to_7.csv"
    )
    extended_path_connection_matrix_8 = os.path.join(
        output_dir_path, "extended_combinations_8.csv"
    )
    extended_path_connection_matrix_9 = os.path.join(
        output_dir_path, "extended_combinations_9.csv"
    )
    probabilities_output_path_connection_matrix_8 = os.path.join(
        output_dir_path, "alternative_probabilities_8.csv"
    )
    probabilities_output_path_connection_matrix_9 = os.path.join(
        output_dir_path, "alternative_probabilities_9.csv"
    )

    alternative_groups = {
        1: [1, 2, 3],
        2: [1, 2, 3, 4, 5],
        3: [1, 2, 3, 4],
        4: [1, 2, 3, 4],
        5: [1, 2, 3, 4],
        6: [1, 2],
        7: [1, 2, 3],
    }

    probabilities_dict = read_probabilities(starting_alternatives_path)
    pair_table_values = read_pair_values(interrelationship_matrix_path)
    connection_matrix_8_data = read_pair_values(connection_matrix_8_path)
    connection_matrix_9_data = read_pair_values(connection_matrix_9_path)

    pc_combinations_results = calculate_pc_combinations(
        alternative_groups, probabilities_dict, pair_table_values
    )

    save_combinations_with_p(pc_combinations_results, comb_pc_path)

    calculate_alternative_probabilities(pc_combinations_results, alt_prob_path)

    combinations_df = pd.read_csv(comb_pc_path)
    process_data(
        combinations_df, connection_matrix_8_data, extended_path_connection_matrix_8
    )

    normalize_and_weight_columns(
        extended_path_connection_matrix_8,
        probabilities_output_path_connection_matrix_8,
        "8.",
    )

    process_data(
        combinations_df, connection_matrix_9_data, extended_path_connection_matrix_9
    )

    normalize_and_weight_columns(
        extended_path_connection_matrix_9,
        probabilities_output_path_connection_matrix_9,
        "9.",
    )
