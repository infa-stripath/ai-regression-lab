import pandas as pd
from pathlib import Path

RESULTS_PATH = Path("output/evaluation_results.csv")
OUTPUT_DIR = Path("output")
THRESHOLD = 4.0


def classify_failures(df):
    df = df.copy()

    # Missing scores are treated as failures.
    df["critical_failure"] = (
        df["critical_failure"].fillna(True).astype(bool)
    )

    df["failed"] = (
        df["overall_score"].isna()
        | (df["overall_score"] < THRESHOLD)
        | df["critical_failure"]
    )

    return df


def main():
    results = pd.read_csv(RESULTS_PATH)

    baseline = results[
        results["version"] == "baseline"
    ].copy()

    candidate = results[
        results["version"] == "candidate"
    ].copy()

    if baseline.empty or candidate.empty:
        raise RuntimeError(
            "Baseline or candidate evaluation results are missing."
        )

    baseline = classify_failures(baseline)
    candidate = classify_failures(candidate)

    baseline = baseline.rename(columns={
        "overall_score": "baseline_score",
        "failed": "baseline_failed",
        "critical_failure": "baseline_critical_failure",
    })

    candidate = candidate.rename(columns={
        "overall_score": "candidate_score",
        "failed": "candidate_failed",
        "critical_failure": "candidate_critical_failure",
    })

    comparison = baseline[
        [
            "case_id",
            "tier",
            "baseline_score",
            "baseline_failed",
            "baseline_critical_failure",
        ]
    ].merge(
        candidate[
            [
                "case_id",
                "candidate_score",
                "candidate_failed",
                "candidate_critical_failure",
                "failure_reason",
            ]
        ],
        on="case_id",
        how="outer",
        indicator=True,
    )

    comparison["score_delta"] = (
        comparison["candidate_score"]
        - comparison["baseline_score"]
    ).round(2)

    comparison["new_failure"] = (
        comparison["candidate_failed"].fillna(True)
        & ~comparison["baseline_failed"].fillna(True)
        & (comparison["_merge"] == "both")
    )

    comparison["status"] = "PASS"

    comparison.loc[
        comparison["candidate_failed"].fillna(True),
        "status",
    ] = "FAIL"

    comparison.loc[
        comparison["_merge"] != "both",
        "status",
    ] = "ERROR: MISSING VERSION"

    comparison.drop(
        columns=["_merge"]
    ).to_csv(
        OUTPUT_DIR / "regression_comparison.csv",
        index=False,
    )

    print("\nRegression comparison:")
    print(
        comparison[
            [
                "case_id",
                "tier",
                "baseline_score",
                "candidate_score",
                "score_delta",
                "new_failure",
                "status",
            ]
        ].to_string(index=False)
    )

    tier_reports = []

    for tier in ["smoke", "critical", "deep"]:
        subset = comparison[comparison["tier"] == tier]

        total = len(subset)
        failures = int(
            subset["candidate_failed"].fillna(True).sum()
        )

        critical_failures = int(
            subset["candidate_critical_failure"].fillna(True).sum()
        )

        pass_rate = (
            (total - failures) / total
            if total else 0
        )

        # Explicit release policy for this lab.
        if tier in ["smoke", "critical"] and failures > 0:
            decision = "FAIL"
        elif tier == "deep" and pass_rate < 0.80:
            decision = "REVIEW"
        elif critical_failures > 0:
            decision = "FAIL"
        else:
            decision = "PASS"

        tier_reports.append({
            "tier": tier,
            "total_cases": total,
            "failed_cases": failures,
            "critical_failures": critical_failures,
            "pass_rate": round(pass_rate, 3),
            "decision": decision,
        })

    tier_df = pd.DataFrame(tier_reports)

    tier_df.to_csv(
        OUTPUT_DIR / "regression_tier_report.csv",
        index=False,
    )

    print("\nTier results:")
    print(tier_df.to_string(index=False))

    if (tier_df["decision"] == "FAIL").any():
        release_decision = "BLOCK RELEASE"
    elif (tier_df["decision"] == "REVIEW").any():
        release_decision = "REQUIRES REVIEW"
    else:
        release_decision = "PASS"

    print("\nFINAL RELEASE DECISION:", release_decision)

    (OUTPUT_DIR / "release_decision.txt").write_text(
        release_decision,
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
