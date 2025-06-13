"""Evaluation module for synthetic data pipeline.

This module provides functions to evaluate synthetic data using descriptive statistics,
performance metrics, and privacy measures.

Functions:
    run_descriptive: Compute basic descriptive statistics and distributions.
    run_performance: Evaluate the performance of synthetic data.
    run_privacy: Assess privacy metrics for synthetic data.
    synth_and_evaluate: Generate synthetic data and evaluate it.
"""

# Import necessary libraries
import random
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd  # Import pandas for DataFrame operations
from sdmetrics.reports.single_table import QualityReport
from sdmetrics.single_column import (
    CategoryCoverage,
    MissingValueSimilarity,
    RangeCoverage,
    StatisticSimilarity,
)
from sdmetrics.single_table.privacy import (
    DCRBaselineProtection,
    DCROverfittingProtection,
    DisclosureProtection,
)
from sdv.metadata.single_table import SingleTableMetadata

from .synth import make_synthetic_housing  # Ensure seed kwarg is accepted


def run_descriptive(df_lookup: pd.DataFrame, df_synth: pd.DataFrame) -> dict:
    """Compute basic descriptive statistics, distributions, correlations, and missing columns.

    Parameters:
        df_lookup (pd.DataFrame): The original dataset.
        df_synth (pd.DataFrame): The synthetic dataset.

    Returns:
        dict: A dictionary containing numeric summaries and frequency distributions.
    """
    numeric_cols = [
        "BOUWJAAR",
        "OPPERVLAKTE",
        "INHOUD",
        "KAVEL",
        "WOZWAARDE",
        "HUISNUMMER",
    ]
    result = {}
    # Numeric summary: convert to numeric, coerce non-parsable to NaN
    real_num = df_lookup[numeric_cols].apply(pd.to_numeric, errors="coerce")
    synth_num = df_synth[numeric_cols].apply(pd.to_numeric, errors="coerce")
    real_stats = real_num.describe().T
    synth_stats = synth_num.describe().T
    result["numeric_summary"] = real_stats.join(
        synth_stats, lsuffix="_real", rsuffix="_synth"
    ).to_dict()
    # Frequency distributions
    freq = {}
    for col in ["GEBRUIKSDOEL", "STATUS_VBO"]:
        real_freq = df_lookup[col].value_counts(normalize=True)
        synth_freq = df_synth[col].value_counts(normalize=True)
        freq[col] = (
            pd.DataFrame({"real": real_freq, "synthetic": synth_freq})
            .fillna(0)
            .to_dict()
        )
    result["frequency"] = freq
    # Correlations: use coerced numeric arrays
    result["correlations"] = {
        "real": real_num.corr().to_dict(),
        "synthetic": synth_num.corr().to_dict(),
    }
    # missing columns
    missing = list(set(df_lookup.columns) - set(df_synth.columns))
    result["missing_columns"] = missing
    return result


def run_performance(df_lookup: pd.DataFrame, df_synth: pd.DataFrame) -> dict:
    """Evaluate the performance of synthetic data.

    Parameters:
        df_lookup (pd.DataFrame): The original dataset.
        df_synth (pd.DataFrame): The synthetic dataset.

    Returns:
        dict: A dictionary containing performance metrics.
    """
    # prepare evaluation subsets (notebook-style drops)
    common_cols = list(set(df_lookup.columns) & set(df_synth.columns))
    # Drop only mapping and synthetic ID columns; retain numeric fields for performance metrics
    drop_cols = [
        "WONINGTYPE",
        "BUURTNAAM",
        "WIJKNAAM",
        "EIGENAARSNAAM",
        "ADRES",
        "BAG_NUMMER",
        "WOZ_NUMMER",
        "GEBRUIKERSCODE",
        "EIGENAARSCODE",
    ]
    df_real = df_lookup[common_cols].drop(
        columns=[c for c in drop_cols if c in common_cols]
    )
    df_synth_common = df_synth[common_cols].drop(
        columns=[c for c in drop_cols if c in common_cols]
    )
    # explicitly coerce known numeric columns to numeric dtype for metadata and metrics
    numeric_cols = [
        "BOUWJAAR",
        "OPPERVLAKTE",
        "INHOUD",
        "KAVEL",
        "WOZWAARDE",
        "HUISNUMMER",
    ]
    df_real[numeric_cols] = df_real[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df_synth_common[numeric_cols] = df_synth_common[numeric_cols].apply(
        pd.to_numeric, errors="coerce"
    )
    # build metadata and generate report
    meta = SingleTableMetadata()
    meta.detect_from_dataframe(data=df_real)
    metadata_dict = meta.to_dict()
    print(f"Metadata: {metadata_dict}")
    report = QualityReport()
    results = report.generate(
        real_data=df_real, synthetic_data=df_synth_common, metadata=metadata_dict
    )
    perf_scores = report.get_properties()
    # extract core scores
    column_shapes_score = float(
        perf_scores.loc[perf_scores.Property == "Column Shapes", "Score"].iloc[0]
    )
    column_pair_trends_score = float(
        perf_scores.loc[perf_scores.Property == "Column Pair Trends", "Score"].iloc[0]
    )

    # single-column
    cat_cols = [c for c in df_real.columns if df_real[c].dtype == "object"]
    # average only valid category-coverage scores
    cat_scores = [
        CategoryCoverage.compute(df_real[c], df_synth_common[c]) for c in cat_cols
    ]
    # Filter out invalid scores (NaN values)
    valid_cat = [s for s in cat_scores if not pd.isna(s)]
    # Calculate the average category coverage score
    category_coverage_score = sum(valid_cat) / len(valid_cat)

    # Compute missing value similarity scores for all columns
    mv_scores = [
        MissingValueSimilarity.compute(df_real[c], df_synth_common[c])
        for c in df_real.columns
    ]
    # Filter out invalid scores (NaN values)
    valid_mv = [s for s in mv_scores if not pd.isna(s)]
    # Calculate the average missing value similarity score
    missing_value_similarity_score = sum(valid_mv) / len(valid_mv)

    # Compute range coverage scores for numeric columns
    rc_scores = [
        RangeCoverage.compute(df_real[c], df_synth_common[c])
        for c in numeric_cols
        if c in df_real.columns and df_synth_common[c].nunique() > 1
    ]
    valid_rc = [s for s in rc_scores if not pd.isna(s)]
    range_coverage_score = sum(valid_rc) / len(valid_rc) if valid_rc else None

    # Compute statistic similarity scores for numeric columns
    ss_scores = [
        StatisticSimilarity.compute(df_real[c], df_synth_common[c])
        for c in numeric_cols
        if c in df_real.columns and df_synth_common[c].nunique() > 1
    ]
    valid_ss = [s for s in ss_scores if not pd.isna(s)]
    statistic_similarity_score = sum(valid_ss) / len(valid_ss) if valid_ss else None

    # Calculate overall performance score based on weights
    overall_performance_score = (
        (0.1 * column_shapes_score)
        + (0.1 * column_pair_trends_score)
        + (0.2 * category_coverage_score)
        + (0.2 * range_coverage_score if range_coverage_score is not None else 0)
        + (0.1 * missing_value_similarity_score)
        + (
            0.3 * statistic_similarity_score
            if statistic_similarity_score is not None
            else 0
        )
    )

    return {
        "general_score": overall_performance_score,
        "column_shapes": column_shapes_score,
        "pair_trends": column_pair_trends_score,
        "category_coverage": category_coverage_score,
        "missing_value_similarity": missing_value_similarity_score,
        "range_coverage": range_coverage_score,
        "statistic_similarity": statistic_similarity_score,
    }


def run_privacy(df_lookup: pd.DataFrame, df_synth: pd.DataFrame) -> dict:
    """Assess privacy metrics for synthetic data.

    Parameters:
        df_lookup (pd.DataFrame): The original dataset.
        df_synth (pd.DataFrame): The synthetic dataset.

    Returns:
        dict: A dictionary containing privacy metrics.
    """
    # drop mapping columns
    drop_cols = [
        "WONINGTYPE",
        "BUURTNAAM",
        "WIJKNAAM",
        "WONINGTYPE_",
        "BOUWJAAR_",
        "WOZWAARDE_",
        "OPPERVLAKTE_",
        "ADRES",
    ]
    common = list(set(df_lookup.columns) & set(df_synth.columns))
    df_real = df_lookup[common].drop(columns=[c for c in drop_cols if c in common])
    df_synth_eval = df_synth[common].drop(columns=[c for c in drop_cols if c in common])

    # Compute overfitting protection with multiple iterations
    scores_DCRo = []
    for i in range(30):
        # Generate train-validation split
        df_real_train = df_real.sample(frac=0.6, random_state=i)
        df_validation_eval = df_real.drop(df_real_train.index)

        # Generate metadata
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data=df_real_train)
        metadata_dict = metadata.to_dict()

        # Compute DCROverfittingProtection
        score = DCROverfittingProtection.compute(
            real_training_data=df_real_train,
            real_validation_data=df_validation_eval,
            synthetic_data=df_synth_eval,
            metadata=metadata_dict,
        )
        scores_DCRo.append(score)

    # Average DCRO scores
    score_DCRo = sum(scores_DCRo) / len(scores_DCRo) if scores_DCRo else 0

    # Compute baseline protection
    meta_b = SingleTableMetadata()
    meta_b.detect_from_dataframe(data=df_real)
    dcr_b = DCRBaselineProtection.compute(
        real_data=df_real,
        synthetic_data=df_synth_eval,
        metadata=meta_b.to_dict(),
    )

    # Iterated disclosure
    dp_scores = []
    sens = ["GEBRUIKERSCODE", "EIGENAARSCODE"]
    others = [c for c in df_real.columns if c not in sens]
    for i in range(100):
        known = random.sample(others, min(len(others), random.randint(2, 4)))
        dp_scores.append(
            DisclosureProtection.compute(
                real_data=df_real,
                synthetic_data=df_synth_eval,
                known_column_names=known,
                sensitive_column_names=sens,
            )
        )

    # Handle case with no valid disclosure protection scores
    valid_dp = [s for s in dp_scores if not pd.isna(s)]
    dp_avg = (sum(valid_dp) / len(valid_dp)) if valid_dp else 0

    # Calculate weighted average privacy score
    privacy_score = (
        (0.5 * dp_avg) + (0.2 * score_DCRo) + (0.3 * dcr_b if dcr_b is not None else 0)
    )

    return {
        "dcr_overfitting": score_DCRo,
        "dcr_baseline": dcr_b,
        "disclosure": dp_avg,
        "privacy_score": privacy_score,
    }


def synth_and_evaluate(
    table, evaluation_type: str, num_records: int, seed: int
) -> dict:
    """Generate synthetic data and evaluate it.

    Parameters:
        table: The table to synthesize.
        evaluation_type (str): The type of evaluation ('descriptive', 'performance', 'privacy').
        num_records (int): Number of synthetic records to generate.
        seed (int): Random seed for reproducibility.

    Returns:
        dict: A dictionary containing evaluation results.
    """
    df_lookup = pd.DataFrame(table.rows, columns=table.columns)
    df_synth = make_synthetic_housing(num_records, df_lookup, seed=seed)
    df_real = df_lookup
    if evaluation_type == "descriptive":
        return run_descriptive(df_real, df_synth)
    if evaluation_type == "performance":
        return run_performance(df_real, df_synth)
    if evaluation_type == "privacy":
        return run_privacy(df_real, df_synth)
    raise ValueError("Unknown evaluation type")
