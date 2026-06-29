from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


TIME_COL = "时间"
TRUE_COL = "True_Step1_KW"
PRED_COL = "Pred_Step1_KW"
ERROR_COL = "Error_KW"
ABS_ERROR_COL = "Abs_Error_KW"
APE_COL = "APE"


def list_datasets(data_dir: str | Path = "data") -> list[str]:
    input_dir = Path(data_dir) / "INPUT"
    datasets = [
        path.name.replace("_10min_top10_features.csv", "")
        for path in input_dir.glob("*_10min_top10_features.csv")
    ]
    return sorted(datasets)


def load_prediction(dataset: str, data_dir: str | Path = "data") -> pd.DataFrame:
    path = Path(data_dir) / "PRED_RESULT" / f"{dataset}_step1_true_pred_inverse.csv"
    df = pd.read_csv(path)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    if ERROR_COL not in df.columns:
        df[ERROR_COL] = df[PRED_COL] - df[TRUE_COL]
    if ABS_ERROR_COL not in df.columns:
        df[ABS_ERROR_COL] = df[ERROR_COL].abs()
    if "Squared_Error_KW" not in df.columns:
        df["Squared_Error_KW"] = df[ERROR_COL] ** 2
    return df


def load_features(dataset: str, data_dir: str | Path = "data") -> pd.DataFrame:
    path = Path(data_dir) / "INPUT" / f"{dataset}_10min_top10_features.csv"
    df = pd.read_csv(path)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    return df


def load_metrics(data_dir: str | Path = "data") -> pd.DataFrame:
    metric_dir = Path(data_dir) / "METRIC_RESULT"
    frames = [pd.read_csv(path) for path in sorted(metric_dir.glob("*.csv"))]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    return df.rename(columns={"Model_Name": "dataset"})


def load_loss(dataset: str, data_dir: str | Path = "data") -> pd.DataFrame:
    path = Path(data_dir) / "LOSS_DATA" / f"{dataset}_loss.csv"
    df = pd.read_csv(path)
    df.insert(0, "epoch", np.arange(1, len(df) + 1))
    return df


def load_merged_dataset(dataset: str, data_dir: str | Path = "data") -> pd.DataFrame:
    features = load_features(dataset, data_dir)
    prediction = load_prediction(dataset, data_dir)
    merged = prediction.merge(features, on=TIME_COL, how="inner")
    merged.insert(0, "dataset", dataset)
    return merged


def input_feature_columns(df: pd.DataFrame) -> list[str]:
    reserved = {
        "dataset",
        TIME_COL,
        TRUE_COL,
        PRED_COL,
        ERROR_COL,
        ABS_ERROR_COL,
        APE_COL,
        "Squared_Error_KW",
    }
    return [
        column
        for column in df.columns
        if column not in reserved and pd.api.types.is_numeric_dtype(df[column])
    ]


def dataset_summary(data_dir: str | Path = "data") -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    metrics = load_metrics(data_dir)
    metrics_by_name = (
        metrics.set_index("dataset").to_dict("index") if not metrics.empty else {}
    )

    for dataset in list_datasets(data_dir):
        prediction = load_prediction(dataset, data_dir)
        features = load_features(dataset, data_dir)
        metric_row = metrics_by_name.get(dataset, {})
        rows.append(
            {
                "dataset": dataset,
                "data_rows": len(features),
                "prediction_rows": len(prediction),
                "feature_rows": len(features),
                "start_time": prediction[TIME_COL].min(),
                "end_time": prediction[TIME_COL].max(),
                "RMSE_KW": metric_row.get("RMSE_KW", rmse(prediction[ERROR_COL])),
                "MAE_KW": metric_row.get("MAE_KW", prediction[ABS_ERROR_COL].mean()),
                "R2": metric_row.get("R2"),
                "MAPE": metric_row.get("MAPE", prediction[APE_COL].mean()),
                "input_columns": len(input_feature_columns(features)),
            }
        )
    return pd.DataFrame(rows)


def rmse(error_series: pd.Series) -> float:
    return float(np.sqrt(np.mean(np.square(error_series))))


def downsample_frame(df: pd.DataFrame, max_rows: int = 1500) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df
    step = int(np.ceil(len(df) / max_rows))
    return df.iloc[::step].copy()


def feature_error_correlations(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for column in input_feature_columns(df):
        abs_corr = df[column].corr(df[ABS_ERROR_COL], method="spearman")
        signed_corr = df[column].corr(df[ERROR_COL], method="spearman")
        rows.append(
            {
                "feature": column,
                "spearman_abs_error": abs_corr,
                "spearman_error": signed_corr,
                "abs_spearman_abs_error": abs(abs_corr) if pd.notna(abs_corr) else np.nan,
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values("abs_spearman_abs_error", ascending=False)


def high_error_rows(df: pd.DataFrame, count: int = 10) -> pd.DataFrame:
    columns = [TIME_COL, TRUE_COL, PRED_COL, ERROR_COL, ABS_ERROR_COL, APE_COL]
    return df.nlargest(count, ABS_ERROR_COL)[columns]


def suggested_error_questions(dataset: str, df: pd.DataFrame, count: int = 5) -> list[str]:
    top_errors = high_error_rows(df, count=count)
    corr = feature_error_correlations(df).head(3)
    feature_text = "、".join(corr["feature"].tolist()) if not corr.empty else "主要输入变量"
    questions = [
        f"请解释 {dataset} 中误差最大的几个时间点为什么可能出现较大预测偏差。",
        f"请结合 {feature_text} 与绝对误差的 Spearman 相关性，分析该数据集的误差来源。",
        f"请根据 RMSE、MAE 和高误差样本，判断该模型在哪些工况下可能不稳定。",
    ]

    for _, row in top_errors.iterrows():
        questions.append(
            f"请重点解释 {row[TIME_COL]} 这个时间点的预测误差："
            f"真实值 {row[TRUE_COL]:.3f} kW，预测值 {row[PRED_COL]:.3f} kW，"
            f"绝对误差 {row[ABS_ERROR_COL]:.3f} kW。"
        )
    return questions


def make_error_summary(dataset: str, df: pd.DataFrame) -> str:
    corr = feature_error_correlations(df).head(5)
    top_errors = high_error_rows(df, count=5)
    lines = [
        f"数据集：{dataset}",
        f"样本数：{len(df)}",
        f"时间范围：{df[TIME_COL].min()} 至 {df[TIME_COL].max()}",
        f"RMSE：{rmse(df[ERROR_COL]):.3f} kW",
        f"MAE：{df[ABS_ERROR_COL].mean():.3f} kW",
        f"最大绝对误差：{df[ABS_ERROR_COL].max():.3f} kW",
        "",
        "绝对误差相关性较高的输入变量：",
    ]
    for _, row in corr.iterrows():
        lines.append(f"- {row['feature']}：Spearman 相关系数 {row['spearman_abs_error']:.3f}")
    lines.append("")
    lines.append("绝对误差最大的时间点：")
    for _, row in top_errors.iterrows():
        lines.append(
            "- "
            f"{row[TIME_COL]}，真实值 {row[TRUE_COL]:.3f} kW，"
            f"预测值 {row[PRED_COL]:.3f} kW，绝对误差 {row[ABS_ERROR_COL]:.3f} kW"
        )
    return "\n".join(lines)
