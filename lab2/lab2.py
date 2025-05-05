import os
import csv
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
    all_combinations = []

    def backtrack(index=0, current=[]):
        if index == len(group_keys):
            all_combinations.append(tuple(current))
            return
        group = group_keys[index]
        for item in groups[group]:
            backtrack(index + 1, current + [f"{group}.{item}."])

    backtrack()
    return all_combinations


def compute_pair_product(combo, pair_table):
    product = 1.0
    for i in range(len(combo)):
        for j in range(i + 1, len(combo)):
            a, b = combo[i], combo[j]
            val = None

            if b in pair_table.index and a in pair_table.columns:
                val = pair_table.at[b, a]
                if pd.isna(val):
                    val = None

            if val is None:
                raise KeyError(f"Missing value for pair ({a}, {b})")

            product *= float(val) + 1
    return product


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    table1_path = os.path.join(base_dir, "table1.csv")
    table2_path = os.path.join(base_dir, "table2.csv")
    output_path = os.path.join(base_dir, "combinations_with_p.csv")

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

    prob_dict = read_probabilities(table1_path)
    pair_table = read_pair_values(table2_path)
    combinations = generate_combinations(groups, group_keys)

    results = []
    for combo in combinations:
        try:
            P = 1.0
            for key in combo:
                if key not in prob_dict:
                    raise KeyError(f"Missing probability for {key}")
                P *= float(prob_dict[key])
            pair_product = compute_pair_product(combo, pair_table)
            final_result = P * pair_product
            results.append(list(combo) + [P, pair_product, final_result])
        except Exception as e:
            print(f"Skipping combination {combo} due to error: {e}")
            continue

    # Normalize the FinalResult column
    final_results = [row[-1] for row in results]  # Extracting all FinalResult values
    sum_final_results = sum(final_results)
    normalized_results = [
        final_result / sum_final_results if sum_final_results != 0 else 0
        for final_result in final_results
    ]

    # Add normalized final results to the output
    for idx, normalized in enumerate(normalized_results):
        results[idx].append(normalized)

    # Write the results to a CSV file
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        headers = [f"Group{i}" for i in range(1, 8)] + [
            "P",
            "PairProduct",
            "FinalResult",
            "NormalizedFinalResult",
        ]
        writer.writerow(headers)
        writer.writerows(results)

    print(f"Saved {len(results)} combinations to {output_path}")
