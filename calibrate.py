import pandas as pd

RESULTS_PATH = "output/evaluation_results.csv"
DATASET_PATH = "evaluation_cases.csv"


def calculate_metrics(actual, predicted):
    tp = int(((actual == 1) & (predicted == 1)).sum())
    tn = int(((actual == 0) & (predicted == 0)).sum())
    fp = int(((actual == 0) & (predicted == 1)).sum())
    fn = int(((actual == 1) & (predicted == 0)).sum())

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0

    return {
        "accuracy": round(accuracy, 3),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "false_passes": fp,
        "false_failures": fn,
    }


def main():
    results = pd.read_csv(RESULTS_PATH)
    cases = pd.read_csv(DATASET_PATH)

    candidate = results[
        results["version"] == "candidate"
    ].copy()

    data = candidate.merge(
        cases[["case_id", "human_score"]],
        on="case_id",
        how="inner",
    )

    data = data.dropna(
        subset=["overall_score", "human_score"]
    )

    # Human-reviewed pass/fail labels.
    actual = (data["human_score"] >= 4).astype(int)

    rows = []

    for threshold in [3.0, 3.5, 4.0, 4.5]:
        predicted = (
            (data["overall_score"] >= threshold)
            & (~data["critical_failure"].fillna(True).astype(bool))
        ).astype(int)

        metrics = calculate_metrics(actual, predicted)

        rows.append({
            "threshold": threshold,
            **metrics,
        })

    report = pd.DataFrame(rows)

    print("\nThreshold calibration results:")
    print(report.to_string(index=False))

    report.to_csv(
        "output/threshold_calibration.csv",
        index=False,
    )

    print("\nSaved output/threshold_calibration.csv")


if __name__ == "__main__":
    main()
