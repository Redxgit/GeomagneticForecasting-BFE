import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import numpy as np
import os
import metrics
from cdasws.datarepresentation import DataRepresentation
from spacepy import coordinates as coord
import natsort

COLORS = {
    "red": "\x1b[31m",
    "green": "\x1b[32m",
    "reset": "\x1b[0m",
    "yellow": "\x1B[33m",
    "blue": "\x1B[34m",
}

SUBSET_TO_COLOR = {
    "Train": "green",
    "Validation": "yellow",
    "Test": "blue",
    "Test_key_parameters": "purple",
}

# SYM-H thresholds for storm intensity levels
SYM_H_THRESHOLD_LOW = -90
SYM_H_THRESHOLD_MODERATE = -130
SYM_H_THRESHOLD_INTENSE = -230
SYM_H_THRESHOLD_SUPERINTENSE = -390

COLOR_SUPERINTENSE = "darkmagenta"
COLOR_INTENSE = "firebrick"
COLOR_MODERATE = "goldenrod"
COLOR_LOW = "yellow"
COLOR_INACTIVE = "olivedrab"

SECT_NAMES = ["Global", "Inactive", "Low", "Moderate", "Intense", "Superintense"]
METRICS = ["RMSE", "R2"]
SUMMARY_COLUMNS = ["Storm index", "Subset", "BFE"] + [
    f"{metric} {sect_name}" for metric in METRICS for sect_name in SECT_NAMES
]

"""
def evaluate_sym_metrics(
    df, indices, plot_graphs=True, save_graphs=False, subsets="all", label_name="SYM_H"
):
    sum_sym = []
    all_preds = []

    print_colored("Evaluating for SYM-H index", color="green")
    if subsets == "test":
        subset_data = [
            (storm_dates.TEST_STORMS, "Test storms"),
        ]
    elif subsets == "test_key":
        subset_data = [
            (storm_dates.TEST_KEY_STORMS, "Test key storms"),
        ]
    else:
        subset_data = [
            (storm_dates.TEST_STORMS, "Test storms"),
            (storm_dates.TEST_KEY_STORMS, "Test key storms"),
        ]

    for subset, subset_name in subset_data:
        print_colored(f"Evaluating {subset_name}", color="green")
        for start_date, end_date, storm_index, checksum in subset:
            sd = pd.to_datetime(start_date)
            ed = pd.to_datetime(end_date)

            storm_eval = df[sd:ed]
            labels = indices[sd:ed][[label_name]].copy()

            assert sd == labels.index[0]
            assert ed == labels.index[-1]
            assert labels[label_name].sum() == checksum

            if len(storm_eval) == len(labels):
                storm = labels.join(storm_eval)
                all_preds.append(storm[sd:ed])
                if plot_graphs:
                    plot_prediction_sym(
                        storm,
                        lookahead=1,
                        storm_index=storm_index,
                        save_img=save_graphs,
                    )
                sum_sym.append(
                    evaluate_storm_sym(
                        storm, lookahead=1, storm_index=storm_index, subset=subset_name
                    )
                )
            elif len(storm_eval) == 0:
                print_colored(
                    f"Test storm {storm_index} from {sd} to {ed} is missing, 0 samples found",
                    "red",
                )
                print_colored(
                    f"\texpected samples: {len(labels)}, found samples: {len(storm_eval)}",
                    "red",
                )
            else:
                storm = labels.join(storm_eval)
                all_preds.append(storm[sd:ed])
                print_colored(
                    f"Test storm {storm_index} from {sd} to {ed} is partilly missing",
                    "yellow",
                )
                print_colored(
                    f"\texpected samples: {len(labels)}, found samples: {len(storm_eval)}",
                    "yellow",
                )

    sum_sym = pd.concat(sum_sym)
    sum_sym = pd.concat(
        [
            sum_sym,
            pd.DataFrame(
                data=[
                    [
                        "Mean:",
                        "-",
                        np.round(np.nanmean(sum_sym[sum_sym.columns[2]]), 3),
                        np.round(np.nanmean(sum_sym[sum_sym.columns[3]]), 3),
                        np.round(np.nanmean(sum_sym[sum_sym.columns[4]]), 3),
                        np.round(np.nanmean(sum_sym[sum_sym.columns[5]]), 3),
                        np.round(np.nanmean(sum_sym[sum_sym.columns[6]]), 3),
                        np.round(np.nanmean(sum_sym[sum_sym.columns[7]]), 3),
                        np.round(np.nanmean(sum_sym[sum_sym.columns[8]]), 3),
                        np.round(np.nanmean(sum_sym[sum_sym.columns[9]]), 3),
                        np.round(np.nanmean(sum_sym[sum_sym.columns[10]]), 3),
                        np.round(np.nanmean(sum_sym[sum_sym.columns[11]]), 3),
                    ]
                ],
                columns=[
                    sum_sym.columns[0],
                    sum_sym.columns[1],
                    sum_sym.columns[2],
                    sum_sym.columns[3],
                    sum_sym.columns[4],
                    sum_sym.columns[5],
                    sum_sym.columns[6],
                    sum_sym.columns[7],
                    sum_sym.columns[8],
                    sum_sym.columns[9],
                    sum_sym.columns[10],
                    sum_sym.columns[11],
                ],
            ),
        ],
        ignore_index=True,
    )
    all_preds = pd.concat(all_preds)
    return all_preds, sum_sym
"""


def print_colored(str_to_print, color):
    print(f'{COLORS[color]}{str_to_print}{COLORS["reset"]}')


def from_min_sym_to_class(val):
    """
    Translate the SYM-H peak to the corresponding class according to the TN3000
    """
    if val <= SYM_H_THRESHOLD_SUPERINTENSE:
        return "Superintense"
    elif val <= SYM_H_THRESHOLD_INTENSE:
        return "Intense"
    elif val <= SYM_H_THRESHOLD_MODERATE:
        return "Moderate"
    elif val <= SYM_H_THRESHOLD_LOW:
        return "Low"
    else:
        return "Check if this is a storm"


def plot_prediction_sym(
    dfx, lookahead, storm_index, save_img=True, error_same_y=True, ax=None
):
    df = dfx.copy()
    og_col = df.columns[0]
    pred_col = df.columns[1]
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    lin_pred = df[pred_col].plot(
        ax=ax, color="blue", label=f"{lookahead} hour/s forecast SYM-H", alpha=0.8
    )
    lin_og = df[og_col].plot(ax=ax, color="green", label="Observed SYM-H", alpha=0.8)
    ax.grid(True, linestyle="--", axis="both", which="both", linewidth=1)
    ax.tick_params(axis="y", color="black", labelsize=10, width=3, length=5)
    ax.tick_params(
        axis="x", color="black", which="both", labelsize=10, width=3, length=5
    )
    ax.set_ylabel("SYM-H (nT)", fontsize=15)
    mse = np.round(
        metrics.msem(df.dropna()[pred_col], df.dropna()[og_col], squared=False), 3
    )
    r2_score = np.round(metrics.r2m(df.dropna()[pred_col], df.dropna()[og_col]), 3)
    bfe = np.round(metrics.calculate_BFE(df[og_col], df[pred_col]))
    ax.set_title(
        f"BFE: {bfe:.3f} | Average RMSE: {mse:.3f} | R2: {r2_score:.3f}", fontsize=15
    )
    ax.set_xlabel("")

    twin = ax.twinx()

    df["diff"] = df[pred_col] - df[og_col]
    df["diff"].plot(
        ax=twin, color="red", linewidth=0.6, alpha=0.75, label="Prediction Error"
    )

    ax.plot(0, 0, color="red", label="Prediction Error")

    leg = ax.legend(
        bbox_to_anchor=(0.5, 1.2),
        loc="upper center",
        ncol=3,
        fancybox=True,
        prop={"size": 12},
    )
    leg.get_frame().set_edgecolor("black")
    plt.setp(ax.spines.values(), lw=2, color="black", alpha=1)

    min_sym = df[og_col].min()
    if min_sym <= SYM_H_THRESHOLD_LOW:
        ax.axhline(SYM_H_THRESHOLD_LOW, linestyle="--", color=COLOR_LOW)
        if min_sym <= SYM_H_THRESHOLD_MODERATE:
            ax.axhline(SYM_H_THRESHOLD_MODERATE, linestyle="--", color=COLOR_MODERATE)
            if min_sym <= SYM_H_THRESHOLD_INTENSE:
                ax.axhline(SYM_H_THRESHOLD_INTENSE, linestyle="--", color=COLOR_INTENSE)
                if min_sym <= SYM_H_THRESHOLD_SUPERINTENSE:
                    ax.axhline(
                        SYM_H_THRESHOLD_SUPERINTENSE,
                        linestyle="--",
                        color="black",
                    )

    if df[pred_col].isnull().values.any():
        ax.fill_between(
            df.index,
            min(df[[og_col, pred_col]].min()),
            max(df[[og_col, pred_col]].max()),
            where=np.isnan(df[pred_col]),
            color="gray",
            alpha=0.5,
            hatch="x",
        )

    if error_same_y:
        min_lim = min(ax.get_ylim()[0], twin.get_ylim()[0])
        max_lim = max(ax.get_ylim()[1], twin.get_ylim()[1])
        twin.set_ylim((min_lim, max_lim))
        ax.set_ylim((min_lim, max_lim))
        twin.yaxis.set_ticklabels([])
    else:
        twin.set_ylabel("Prediction Error (nT)", color="red", fontsize=15)
        twin.tick_params(axis="y", colors="red", labelsize=10, width=3, length=5)
        twin.grid(
            True, color="red", linestyle=":", axis="y", which="both", linewidth=0.5
        )

    if save_img:
        fig.savefig(os.path.join("./plots", f"sym_storm_{storm_index}.png"), dpi=400)
    return ax


def evaluate_storm_sym(df, lookahead, storm_index, subset):
    og_col = df.columns[0]
    pred_col = df.columns[1]

    inactive_sect = df.loc[df[og_col] >= SYM_H_THRESHOLD_LOW].copy()
    low_sect = df.loc[
        (df[og_col] >= SYM_H_THRESHOLD_MODERATE) & (df[og_col] < SYM_H_THRESHOLD_LOW)
    ].copy()
    moderate_sect = df.loc[
        (df[og_col] >= SYM_H_THRESHOLD_INTENSE)
        & (df[og_col] < SYM_H_THRESHOLD_MODERATE)
    ].copy()
    intense_sect = df.loc[
        (df[og_col] >= SYM_H_THRESHOLD_SUPERINTENSE)
        & (df[og_col] < SYM_H_THRESHOLD_INTENSE)
    ].copy()
    superintense_sect = df.loc[df[og_col] < SYM_H_THRESHOLD_SUPERINTENSE].copy()

    global_bfe = metrics.calculate_BFE(df[og_col], df[pred_col])
    global_rmse = np.round(metrics.rmse(df[og_col], df[pred_col]), 3)
    global_r2 = np.round(metrics.r2m(df[og_col], df[pred_col]), 3)
    print(
        f"Evaluation for storm {storm_index}, {lookahead} hours ahead forecast, "
        + f"BFE: {global_bfe} RMSE: {global_rmse} R2: {global_r2}"
    )

    rmse_inactive = np.nan
    rmse_low = np.nan
    rmse_moderate = np.nan
    rmse_intense = np.nan
    rmse_superintense = np.nan

    r2_inactive = np.nan
    r2_low = np.nan
    r2_moderate = np.nan
    r2_intense = np.nan
    r2_superintense = np.nan

    rmse_inactive = np.round(
        metrics.msem(inactive_sect[og_col], inactive_sect[pred_col], squared=False), 3
    )
    r2_inactive = np.round(
        metrics.r2m(inactive_sect[og_col], inactive_sect[pred_col]), 3
    )
    print(
        f"\t{len(inactive_sect)} samples in inactive intensity, inactive RMSE: "
        + f"{rmse_inactive}, inactive R2: {r2_inactive}"
    )

    if len(low_sect) > 0:
        rmse_low = np.round(
            metrics.msem(low_sect[og_col], low_sect[pred_col], squared=False), 3
        )
        r2_low = np.round(metrics.r2m(low_sect[og_col], low_sect[pred_col]), 3)
        print(
            f"\t{len(low_sect)} samples in low intensity, low RMSE: {rmse_low}, low R2: {r2_low}"
        )

        if len(moderate_sect) > 0:
            rmse_moderate = np.round(
                metrics.msem(
                    moderate_sect[og_col], moderate_sect[pred_col], squared=False
                ),
                3,
            )
            r2_moderate = np.round(
                metrics.r2m(moderate_sect[og_col], moderate_sect[pred_col]), 3
            )
            print(
                f"\t{len(moderate_sect)} samples in moderate intensity, moderate RMSE: "
                + f"{rmse_moderate}, moderate R2: {r2_moderate}"
            )

            if len(intense_sect) > 0:
                rmse_intense = np.round(
                    metrics.msem(
                        intense_sect[og_col], intense_sect[pred_col], squared=False
                    ),
                    3,
                )
                r2_intense = np.round(
                    metrics.r2m(intense_sect[og_col], intense_sect[pred_col]), 3
                )
                print(
                    f"\t{len(intense_sect)} samples in intense intensity, intense RMSE: "
                    + f"{rmse_intense}, intense R2: {r2_intense}"
                )

                if len(superintense_sect) > 0:
                    rmse_superintense = np.round(
                        metrics.msem(
                            superintense_sect[og_col],
                            superintense_sect[pred_col],
                            squared=False,
                        ),
                        3,
                    )
                    r2_superintense = np.round(
                        metrics.r2m(
                            superintense_sect[og_col], superintense_sect[pred_col]
                        ),
                        3,
                    )
                    print(
                        f"\t{len(superintense_sect)} samples in superintense intensity, "
                        + f"superintense RMSE: {rmse_superintense}, superintense R2: {r2_superintense}"
                    )

    sum_df = pd.DataFrame(
        [
            storm_index,
            subset,
            global_bfe,
            global_rmse,
            rmse_inactive,
            rmse_low,
            rmse_moderate,
            rmse_intense,
            rmse_superintense,
            global_r2,
            r2_inactive,
            r2_low,
            r2_moderate,
            r2_intense,
            r2_superintense,
        ]
    ).T
    sum_df.columns = SUMMARY_COLUMNS
    sum_df[SUMMARY_COLUMNS[0]] = sum_df[SUMMARY_COLUMNS[0]].astype(int)
    return sum_df


def labelize_sym(df):
    og_col = df.columns[0]
    pred_col = df.columns[1]
    df[f"{og_col}_label"] = 0
    df[f"{pred_col}_label"] = 0

    df.loc[
        (df[og_col] >= SYM_H_THRESHOLD_MODERATE) & (df[og_col] < SYM_H_THRESHOLD_LOW),
        f"{og_col}_label",
    ] = 1
    df.loc[
        (df[og_col] >= SYM_H_THRESHOLD_INTENSE)
        & (df[og_col] < SYM_H_THRESHOLD_MODERATE),
        f"{og_col}_label",
    ] = 2
    df.loc[
        (df[og_col] >= SYM_H_THRESHOLD_SUPERINTENSE)
        & (df[og_col] < SYM_H_THRESHOLD_INTENSE),
        f"{og_col}_label",
    ] = 3
    df.loc[df[og_col] < SYM_H_THRESHOLD_SUPERINTENSE, f"{og_col}_label"] = 4

    df.loc[
        (df[pred_col] >= SYM_H_THRESHOLD_MODERATE)
        & (df[pred_col] < SYM_H_THRESHOLD_LOW),
        f"{pred_col}_label",
    ] = 1
    df.loc[
        (df[pred_col] >= SYM_H_THRESHOLD_INTENSE)
        & (df[pred_col] < SYM_H_THRESHOLD_MODERATE),
        f"{pred_col}_label",
    ] = 2
    df.loc[
        (df[pred_col] >= SYM_H_THRESHOLD_SUPERINTENSE)
        & (df[pred_col] < SYM_H_THRESHOLD_INTENSE),
        f"{pred_col}_label",
    ] = 3
    df.loc[df[pred_col] < SYM_H_THRESHOLD_SUPERINTENSE, f"{pred_col}_label"] = 4

    return df


def plot_comparison_sym(
    df, df_to_compare, title_df, title_compare, ax=None, binwidth=metrics.BINWIDTH
):
    bfe_value, bfe_count, bfe_plot, min_val, max_val = metrics.get_BFE_data(
        df[df.columns[0]], df[df.columns[1]], binwidth=binwidth
    )

    (
        bfe_comparison_value,
        bfe_comparison_count,
        bfe_comparison_plot,
        min_val_comparison,
        max_val_comparison,
    ) = metrics.get_BFE_data(
        df_to_compare[df_to_compare.columns[0]],
        df_to_compare[df_to_compare.columns[1]],
        binwidth=binwidth,
    )

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(13, 6))

    ax.plot(
        bfe_plot.index,
        bfe_plot.values,
        linestyle="-",
        color="blue",
        label=title_df,
    )
    ax.plot(
        bfe_comparison_plot.index,
        bfe_comparison_plot.values,
        linestyle="-",
        color="gray",
        label=title_compare,
    )

    ax.plot(0, 0, label="Samples count", color="green")
    ax.fill_between(
        bfe_plot.index,
        bfe_plot.values,
        bfe_comparison_plot.values,
        where=bfe_plot.values > bfe_comparison_plot.values,
        alpha=0.2,
        interpolate=True,
        color="gray",
    )

    ax.fill_between(
        bfe_plot.index,
        bfe_plot.values,
        bfe_comparison_plot.values,
        where=bfe_plot.values < bfe_comparison_plot.values,
        alpha=0.2,
        interpolate=True,
        color="blue",
    )

    ax.set_ylim(0, ax.get_ylim()[1])
    ax.tick_params(axis="both", which="major", labelsize=14, width=2, length=10)
    ax.set_xlim(bfe_plot.index.min(), bfe_plot.index.max())
    twin = ax.twinx()
    twin.bar(
        bfe_count.index,
        bfe_count.values,
        align="center",
        width=binwidth,
        alpha=0.15,
        color="green",
    )
    twin.set_yscale("log")
    twin.grid(linestyle="--")
    ax.set_xlabel("SYM-H (nT)", fontsize=18)
    ax.set_ylabel("Mean Absolute Difference (nT)", fontsize=18)

    twin.set_ylabel("Bin count", fontsize=18)
    twin.tick_params(axis="y", which="major", labelsize=14, width=2, length=10)
    twin.tick_params(axis="y", which="minor", width=1, length=5)

    if min_val <= SYM_H_THRESHOLD_LOW:
        ax.axvline(SYM_H_THRESHOLD_LOW, linestyle="--", color=COLOR_LOW)
        if min_val <= SYM_H_THRESHOLD_MODERATE:
            ax.axvline(SYM_H_THRESHOLD_MODERATE, linestyle="--", color=COLOR_MODERATE)
            if min_val <= SYM_H_THRESHOLD_INTENSE:
                ax.axvline(SYM_H_THRESHOLD_INTENSE, linestyle="--", color=COLOR_INTENSE)
                if min_val <= SYM_H_THRESHOLD_SUPERINTENSE:
                    ax.axvline(
                        SYM_H_THRESHOLD_SUPERINTENSE,
                        linestyle="--",
                        color=COLOR_SUPERINTENSE,
                    )

    ax.set_title(
        f"Diff BFE ({title_compare} - {title_df}): {bfe_comparison_value - bfe_value:.3f}",
        fontsize=18,
    )
    leg = ax.legend(
        ncol=4,
        fancybox=True,
        prop={"size": 14},
        bbox_to_anchor=(0.5, 1.2),
        loc="upper center",
    )
    leg.get_lines()[-1].set_linewidth(12.0)
    leg.get_lines()[-1].set_alpha(0.15)
    leg.get_frame().set_edgecolor("black")
    plt.setp(ax.spines.values(), lw=2, color="black", alpha=1)
    twin.grid(False)
    ax.grid(True)
    return ax


def plot_evaluation_sym(df, title, ax=None, binwidth=metrics.BINWIDTH, fontsizes=18):
    bfe_value, bfe_count, bfe_plot, min_val, max_val = metrics.get_BFE_data(
        df[df.columns[0]], df[df.columns[1]], binwidth=binwidth
    )

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(13, 6))

    ax.plot(
        bfe_plot.index,
        bfe_plot.values,
        linestyle="-",
        label="Evaluated predictions",
    )
    ax.fill_between(
        bfe_plot.index,
        bfe_plot.values,
        0,
        alpha=0.2,
        color="blue",
    )
    ax.plot(0, 0, color="green", label="Bin count")
    ax.set_ylim(0, ax.get_ylim()[1])
    ax.tick_params(axis="both", which="major", labelsize=14, width=2, length=10)
    ax.set_xlim(bfe_plot.index.min(), bfe_plot.index.max())
    twin = ax.twinx()
    twin.bar(
        bfe_count.index,
        bfe_count.values,
        align="center",
        width=binwidth,
        alpha=0.15,
        color="green",
    )
    twin.set_yscale("log")
    twin.grid(linestyle="--")
    ax.set_xlabel("SYM-H (nT)", fontsize=fontsizes)
    ax.set_ylabel("Mean Absolute Difference (nT)", fontsize=fontsizes)

    twin.set_ylabel("Bin count", fontsize=fontsizes)
    twin.tick_params(
        axis="y", which="major", labelsize=fontsizes - 4, width=2, length=10
    )
    twin.tick_params(axis="y", which="minor", width=1, length=5)

    if min_val <= SYM_H_THRESHOLD_LOW:
        ax.axvline(SYM_H_THRESHOLD_LOW, linestyle="--", color=COLOR_LOW)
        if min_val <= SYM_H_THRESHOLD_MODERATE:
            ax.axvline(SYM_H_THRESHOLD_MODERATE, linestyle="--", color=COLOR_MODERATE)
            if min_val <= SYM_H_THRESHOLD_INTENSE:
                ax.axvline(SYM_H_THRESHOLD_INTENSE, linestyle="--", color=COLOR_INTENSE)
                if min_val <= SYM_H_THRESHOLD_SUPERINTENSE:
                    ax.axvline(
                        SYM_H_THRESHOLD_SUPERINTENSE,
                        linestyle="--",
                        color=COLOR_SUPERINTENSE,
                    )

    # ax.set_title(f"AUC: {int(auc_total)} | NAUC: {auc_total / sym_range:.3f} | Avg points: {mean_value_to_plot.mean():.3f} | {title}")
    ax.set_title(f"BFE: {bfe_value:.3f} | {title}", fontsize=fontsizes)
    leg = ax.legend(
        ncol=2,
        fancybox=True,
        prop={"size": fontsizes - 4},
        bbox_to_anchor=(0.5, 1.2),
        loc="upper center",
    )
    leg.get_lines()[-1].set_linewidth(12.0)
    leg.get_lines()[-1].set_alpha(0.15)
    leg.get_frame().set_edgecolor("black")
    plt.setp(ax.spines.values(), lw=2, color="black", alpha=1)
    twin.grid(False)
    ax.grid(True)
    return ax


B_MAGNITUDE_COL_NAME = "Bmag"
BX_COL_NAME = "Bx"
BY_COL_NAME = "By"
BZ_COL_NAME = "Bz"
PROTON_SPEED_COL_NAME = "Proton_speed"
PROTON_TEMPERATURE_COL_NAME = "Proton_temp"
PROTON_DENSITY_COL_NAME = "Proton_density"
SYM_H_COL_NAME = "SYM_H"

ALTERNATIVE_NAMES_BMAG = ["Bt", "Magnitude", "F", "B1F1", "bt"]
ALTERNATIVE_NAMES_BX = ["bx_gsm", "Bx (GSM)", "BX_GSE"]
ALTERNATIVE_NAMES_BY = ["by_gsm", "By (GSM)", "BY_GSM"]
ALTERNATIVE_NAMES_BZ = ["bz_gsm", "Bz (GSM)", "BZ_GSM"]
ALTERNATIVE_NAMES_PROTON_DENSITY = [
    "Proton density",
    "density",
    "Np",
    "nH",
    "proton_density",
]
ALTERNATIVE_NAMES_PROTON_SPEED = ["Proton speed", "speed", "Vp", "vH", "flow_speed"]
ALTERNATIVE_NAMES_PROTON_TEMPERATURE = [
    "Temperature",
    "temperature",
    "Tpr",
    "T",
    "THERMAL_TEMP",
]
ALTERNATIVE_NAMES_SYM_H = ["SYM"]

PARSING_DICT = [
    (B_MAGNITUDE_COL_NAME, ALTERNATIVE_NAMES_BMAG),
    (BX_COL_NAME, ALTERNATIVE_NAMES_BX),
    (BY_COL_NAME, ALTERNATIVE_NAMES_BY),
    (BZ_COL_NAME, ALTERNATIVE_NAMES_BZ),
    (PROTON_DENSITY_COL_NAME, ALTERNATIVE_NAMES_PROTON_DENSITY),
    (PROTON_TEMPERATURE_COL_NAME, ALTERNATIVE_NAMES_PROTON_TEMPERATURE),
    (PROTON_SPEED_COL_NAME, ALTERNATIVE_NAMES_PROTON_SPEED),
    (SYM_H_COL_NAME, ALTERNATIVE_NAMES_SYM_H),
]


def parse_column_names(df):
    for col in df.columns:
        for col_name, col_alternatives in PARSING_DICT:
            if col in col_alternatives:
                df = df.rename(columns={col: col_name})
                continue

    return df


def plot_storm_sym(df, subset="Test", isnew="True", storm_index=1):
    fig, ax = plt.subplots(figsize=(10, 5))

    col = "green" if isnew else "blue"
    if subset == "Test_key_parameters":
        col = "purple"

    df["SYM_H"].plot(legend=False, xlabel="Date", ax=ax, color=col)

    # Set the y-axis label
    ax.set_ylabel("SYM-H (nT)", fontsize=12)

    # Add horizontal lines to represent different SYM-H thresholds for storm intensity levels
    min_sym = df["SYM_H"].min()

    if min_sym <= SYM_H_THRESHOLD_LOW:
        ax.axhline(SYM_H_THRESHOLD_LOW, linestyle="--", color=COLOR_LOW)

        if min_sym <= SYM_H_THRESHOLD_MODERATE:
            ax.axhline(SYM_H_THRESHOLD_MODERATE, linestyle="--", color=COLOR_MODERATE)

            if min_sym <= SYM_H_THRESHOLD_INTENSE:
                ax.axhline(SYM_H_THRESHOLD_INTENSE, linestyle="--", color=COLOR_INTENSE)

                if min_sym <= SYM_H_THRESHOLD_SUPERINTENSE:
                    ax.axhline(
                        SYM_H_THRESHOLD_SUPERINTENSE,
                        linestyle="--",
                        color=COLOR_SUPERINTENSE,
                    )

    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:>8.0f}"))
    add_str = "New" if isnew else ""
    ax.set_title(
        f'{add_str} {subset} Storm {storm_index} from {df.index[0].strftime("%Y-%m-%d")} to {df.index[-1].strftime("%Y-%m-%d")} Min SYM-H: {int(min_sym)} nT | class: {from_min_sym_to_class(min_sym)}',
        fontsize=13,
    )

    return ax


def superposed_plot_sym(
    df,
    train_indices,
    val_indices,
    test_indices,
    test_key_indices,
    names,
    title,
    ax=None,
):
    all_indices = [train_indices, val_indices, test_indices, test_key_indices]

    delta_backward = pd.DateOffset(days=5)
    delta_forward = pd.DateOffset(days=5)

    superposed_total = pd.DataFrame(
        index=pd.timedelta_range(start=f"-2 days", end=f"4 days", freq="5min")
    )
    superposed_dfs_total = []

    min_sym = np.nanmin(
        [
            train_indices["Min SYM-H"].min(),
            val_indices["Min SYM-H"].min(),
            test_indices["Min SYM-H"].min(),
            test_key_indices["Min SYM-H"].min(),
        ]
    )

    for i in range(len(all_indices)):
        subset_data = all_indices[i]
        name = names[i]
        superposed_df = pd.DataFrame(
            index=pd.timedelta_range(start=f"-2 days", end=f"4 days", freq="5min")
        )

        print(f"Evaluating for indices: {name} | total storms: {len(subset_data)}")

        if len(subset_data) == 0:
            continue

        dfs_tmp = []

        for storm_ind, storm in subset_data.iterrows():
            # Get data for the current storm
            strm = df[
                storm["Start date"] - delta_backward : storm["End date"] + delta_forward
            ]
            strm = strm[["SYM_H"]]
            idxmin = strm["SYM_H"].idxmin()

            strm.columns = [[f"storm-{storm_ind}"]]
            strm = strm[idxmin - delta_backward : idxmin + delta_forward].copy()

            # Create a new index for the data in 5-minute intervals
            strm.index = pd.timedelta_range(
                start=f"-5 days", freq="5min", periods=len(strm)
            )

            dfs_tmp.append(strm)

            superposed_df = pd.concat(dfs_tmp, axis=1)

        superposed_dfs_total.append(superposed_df.copy())
        # Average the storms for the superposed epoch plot
        superposed_df[f"{name} Superposed"] = superposed_df.mean(axis=1)

        superposed_total[f"{name} Superposed"] = superposed_df[f"{name} Superposed"]

    # Create a new column in the main dataframe for hours to ease the plotting

    superposed_total["hours"] = superposed_total.index / pd.Timedelta(hours=1)
    superposed_dfs_total = pd.concat(superposed_dfs_total, axis=1)
    superposed_total["overall_mean"] = superposed_dfs_total.mean(axis=1, skipna=True)

    # Plot the superposed epoch storms for each class
    if ax is None:
        fig, ax = plt.subplots(1, figsize=(8, 4))

    mad_deviation_training = "NA"
    mad_deviation_val = "NA"
    mad_deviation_test = "NA"
    mad_deviation_test_key = "NA"

    if len(train_indices) > 0:
        ax = superposed_total.plot(
            x="hours",
            y=f"{names[0]} Superposed",
            color="green",
            ax=ax,
            alpha=0.8,
            label=f"{names[0]} ({len(train_indices)})",
        )

        mad_deviation_training = np.round(
            (
                superposed_total[f"{names[0]} Superposed"]
                - superposed_total["overall_mean"]
            )
            .abs()
            .mean(),
            3,
        )

    if len(val_indices) > 0:
        ax = superposed_total.plot(
            x="hours",
            y=f"{names[1]} Superposed",
            color="gray",
            ax=ax,
            alpha=0.8,
            label=f"{names[1]} ({len(val_indices)})",
        )

        mad_deviation_val = np.round(
            (
                superposed_total[f"{names[1]} Superposed"]
                - superposed_total["overall_mean"]
            )
            .abs()
            .mean(),
            3,
        )

    if len(test_indices) > 0:
        ax = superposed_total.plot(
            x="hours",
            y=f"{names[2]} Superposed",
            color="blue",
            ax=ax,
            alpha=0.8,
            label=f"{names[2]} ({len(test_indices)})",
        )

        mad_deviation_test = np.round(
            (
                superposed_total[f"{names[2]} Superposed"]
                - superposed_total["overall_mean"]
            )
            .abs()
            .mean(),
            3,
        )

    if len(test_key_indices) > 0:
        ax = superposed_total.plot(
            x="hours",
            y=f"{names[3]} Superposed",
            color="magenta",
            ax=ax,
            alpha=0.8,
            label=f"{names[3]} ({len(test_key_indices)})",
        )

        mad_deviation_test_key = np.round(
            (
                superposed_total[f"{names[3]} Superposed"]
                - superposed_total["overall_mean"]
            )
            .abs()
            .mean(),
            3,
        )

    ax.patch.set_edgecolor("black")
    ax.patch.set_linewidth(1)

    print(
        f"Training MAD: {mad_deviation_training}, Validation MAD: {mad_deviation_val},"
        + f" Test MAD: {mad_deviation_test}, Test key MAD: {mad_deviation_test_key}"
    )

    if min_sym <= SYM_H_THRESHOLD_LOW:
        ax.axhline(SYM_H_THRESHOLD_LOW, linestyle="--", color=COLOR_LOW)
        if min_sym <= SYM_H_THRESHOLD_MODERATE:
            ax.axhline(SYM_H_THRESHOLD_MODERATE, linestyle="--", color=COLOR_MODERATE)
            if min_sym <= SYM_H_THRESHOLD_INTENSE:
                ax.axhline(SYM_H_THRESHOLD_INTENSE, linestyle="--", color=COLOR_INTENSE)
                if min_sym <= SYM_H_THRESHOLD_SUPERINTENSE:
                    ax.axhline(
                        SYM_H_THRESHOLD_SUPERINTENSE,
                        linestyle="--",
                        color=COLOR_SUPERINTENSE,
                    )

    ax.set_ylabel("SYM-H (nT)")
    ax.set_xlabel("Hours around peak")
    ax.set_title(title, fontsize=13)


def convert_to_datetime(date, utc = False):
    """
    Converts a date string to a datetime object.
    """
    formats = ["%Y%m%d", "%Y%m%d%H%M"]
    for fmt in formats:
        try:
            return pd.to_datetime(date, format=fmt, errors="raise", utc = utc)
        except ValueError:
            pass
    raise ValueError("Invalid date format")


def download_data(cdas, database_identifier, columns, start_date, end_date):
    """
    Downloads data from the CDAS (Coordinated Data Analysis System) database using the cadsws package.
    Data will be saved raw (without preprocessing)
    """
    data = cdas.get_data(
        database_identifier,
        columns,
        start_date,
        end_date,
        dataRepresentation=DataRepresentation.XARRAY,
    )[1]

    df = pd.DataFrame(data=data["Epoch"], columns=["datetime"])
    df["datetime"] = pd.to_datetime(df["datetime"])

    # Transform the XArray data representation to csv
    metavar_index = 0
    for var_index, dvar in enumerate(list(data.keys())):
        if data[dvar].ndim == 1:
            if len(data[dvar]) < 10:
                continue
            df[dvar] = np.array(data[dvar])
        else:
            cols = [f"{dvar}_X", f"{dvar}_Y", f"{dvar}_Z"]
            df[cols] = data[dvar].values
            metavar_index += 1

    df.set_index("datetime", inplace=True)
    return df


def preprocess_ace_imf(df):
    df.replace(float(-1.0e31), np.nan, inplace=True)
    df.rename(
        columns={"Magnitude": "Bmag", "BGSM_X": "Bx", "BGSM_Y": "By", "BGSM_Z": "Bz"},
        inplace=True,
    )
    df = df.groupby(pd.Grouper(freq="5min", closed="right", label="right")).mean()
    df.columns = df.columns.str.split("_mean").str[0]
    return df


def preprocess_ace_swepam(df):
    df.replace(float(-1.0e31), np.nan, inplace=True)
    df.rename(
        columns={
            "V_GSM_X": "Proton_speed_x",
            "V_GSM_Y": "Proton_speed_y",
            "V_GSM_Z": "Proton_speed_z",
            "Vp": "Proton_speed",
            "Np": "Proton_density",
            "Tpr": "Proton_temp",
        },
        inplace=True,
    )
    df = df.groupby(pd.Grouper(freq="5min", closed="right", label="right")).mean()
    df.columns = df.columns.str.split("_mean").str[0]
    return df


def calculate_derived_params(df):
    """
    Calculates derived parameters based on existing data columns.
    """
    df["Pressure"] = (2 * 10**-6) * df["Proton_density"] * df["Proton_speed_x"] ** 2
    df["E_field"] = np.abs(df["Proton_speed_x"]) * df["Bz"] * 10**-3
    return df


VALID_RANGES_VARS_ACE_IMF = {
    "Magnitude": (0.0, 500.0),
    "Br RTN": (-100.0, 100.0),
    "Bt RTN": (-100.0, 100.0),
    "Bn RTN": (-100.0, 100.0),
    "Bx GSE": (-100.0, 100.0),
    "By GSE": (-100.0, 100.0),
    "Bz GSE": (-100.0, 100.0),
    "Bx (GSM)": (-100.0, 100.0),
    "By (GSM)": (-100.0, 100.0),
    "Bz (GSM)": (-100.0, 100.0),
    "Bmag": (0.0, 100.0),
    "Bx": (-100.0, 100.0),
    "By": (-100.0, 100.0),
    "Bz": (-100.0, 100.0),
}


def preprocess_ace_imf_provisional(
    df,
    cols=[B_MAGNITUDE_COL_NAME, BX_COL_NAME, BY_COL_NAME, BZ_COL_NAME],
    group_freq="5min",
    group_vars=["mean"],
    group_closed="right",
    group_label="right",
):
    new_cols = []

    for col in df.columns:
        if col.find("[PRELIM] ") >= 0:
            new_cols.append(col.split("[PRELIM] ")[1])
        else:
            new_cols.append(col)

    df.columns = new_cols
    df = parse_column_names(df)

    for col in df.columns:
        if df[col].dtype == "float32":
            for v in (-9999999848243207295109594873856.0, float(-1.0e31), -999.9):
                df[col].replace(v, np.nan, inplace=True)

    # Replace column values which are outside their range
    for col in df.columns:
        if col in VALID_RANGES_VARS_ACE_IMF.keys():
            df.loc[df[col] >= VALID_RANGES_VARS_ACE_IMF[col][1], col] = np.nan
            df.loc[df[col] <= VALID_RANGES_VARS_ACE_IMF[col][0], col] = np.nan

    if ("BGSEc_X") in df.columns:
        gse_to_gsm = coord.Coords(
            df[["BGSEc_X", "BGSEc_Y", "BGSEc_Z"]].values, "GSM", "car", use_irbem=False
        )
        gse_to_gsm = gse_to_gsm.convert("GSM", "car")
        df["Bx"] = gse_to_gsm.x
        df["By"] = gse_to_gsm.y
        df["Bz"] = gse_to_gsm.z

    for col in df.columns:
        if col in VALID_RANGES_VARS_ACE_IMF.keys():
            df.loc[df[col] >= VALID_RANGES_VARS_ACE_IMF[col][1], col] = np.nan
            df.loc[df[col] <= VALID_RANGES_VARS_ACE_IMF[col][0], col] = np.nan

    df = df[cols]
    df = df.groupby(
        pd.Grouper(freq=group_freq, closed=group_closed, label=group_label)
    ).agg(group_vars)
    df.columns = ["_".join(x) for x in df.columns]

    if "mean" in group_vars:
        new_cols = []
        for col in df.columns:
            if col.find("_mean"):
                new_cols.append(col.split("_" + group_vars[0])[0])
            else:
                new_cols.append(col)
        df.columns = new_cols

    return df


VALID_RANGES_VARS_ACE_SWEPAM = {
    "Np": (0.0, 200.0),
    "Vp": (0.0, 2500.0),
    "Tpr": (1000.0, 1100000.0),
    "Proton_density": (0.0, 200.0),
    "Proton_speed": (0.0, 2500.0),
    "Proton_temp": (1000.0, 1100000.0),
    "alpha_ratio": (0.0, 10.0),
    "VX (GSE)": (-2000.0, 0.0),
    "VY (GSE)": (-900.0, 900.0),
    "VZ (GSE)": (-900.0, 900.0),
    "VR (RTN)": (-2000.0, 0.0),
    "VT (RTN)": (-900.0, 900.0),
    "VN (RTN)": (-900.0, 900.0),
    "VX (GSM)": (-1800.0, 0.0),
    "VY (GSM)": (-900.0, 900.0),
    "VZ (GSM)": (-900.0, 900.0),
    "X GSE": (-2000000.0, 2000000.0),
    "Y GSE": (-2000000.0, 2000000.0),
    "Z GSE": (-2000000.0, 2000000.0),
}


def preprocess_ace_swepam_provisional(
    df,
    cols=[PROTON_DENSITY_COL_NAME, PROTON_SPEED_COL_NAME, PROTON_TEMPERATURE_COL_NAME],
    group_freq="5min",
    group_vars=["mean"],
    group_closed="right",
    group_label="right",
):
    new_cols = []

    for col in df.columns:
        if col.find("[PRELIM] ") >= 0:
            new_cols.append(col.split("[PRELIM] ")[1])
        else:
            new_cols.append(col)

    df.columns = new_cols
    df = parse_column_names(df)

    for col in df.columns:
        if df[col].dtype == "float32":
            for v in (-9999999848243207295109594873856.0, float(-1.0e31)):
                df[col].replace(v, np.nan, inplace=True)

    # Replace column values which are outside their range
    for col in df.columns:
        if col in VALID_RANGES_VARS_ACE_SWEPAM.keys():
            df.loc[df[col] >= VALID_RANGES_VARS_ACE_SWEPAM[col][1], col] = np.nan
            df.loc[df[col] <= VALID_RANGES_VARS_ACE_SWEPAM[col][0], col] = np.nan

    df = df[cols]
    df = df.groupby(
        pd.Grouper(freq=group_freq, closed=group_closed, label=group_label)
    ).agg(group_vars)
    df.columns = ["_".join(x) for x in df.columns]

    if "mean" in group_vars:
        new_cols = []
        for col in df.columns:
            if col.find("_mean"):
                new_cols.append(col.split("_" + group_vars[0])[0])
            else:
                new_cols.append(col)
        df.columns = new_cols

    return df

# Database identifiers
DATABASE_ACE_IMF_16 = "AC_H0_MFI"
DATABASE_ACE_SWEPAM = "AC_H0_SWE"
DATABASE_ACE_IMF_PROVISIONAL = "AC_K1_MFI"
DATABASE_ACE_PLASMA_PROVISIONAL = "AC_K0_SWE"
DATABASE_OMNI = "OMNI_HRO_5MIN"

# Variable names
VARS_ACE_IMF_H0 = ["Magnitude", "BGSM"]
VARS_ACE_SWEPAM = ["V_GSM", "Vp", "Np", "Tpr"]
VARS_ACE_IMF_PROVISIONAL = ["Magnitude", "BGSEc"]
VARS_ACE_SWEPAM_PROVISIONAL = ["Np", "Vp", "Tpr"]
VARS_OMNI = ["SYM_H"]

def download_data(cdas, database_identifier, columns, start_date, end_date):
    """
    Downloads data from the CDAS (Coordinated Data Analysis System) database using the cadsws package.
    Data will be saved raw (without preprocessing)
    """
    data = cdas.get_data(
        database_identifier,
        columns,
        start_date,
        end_date,
        dataRepresentation=DataRepresentation.XARRAY,
    )[1]

    df = pd.DataFrame(data=data["Epoch"], columns=["datetime"])
    df["datetime"] = pd.to_datetime(df["datetime"])

    # Transform the XArray data representation to csv
    metavar_index = 0
    for var_index, dvar in enumerate(list(data.keys())):
        if data[dvar].ndim == 1:
            if len(data[dvar]) < 10:
                continue
            df[dvar] = np.array(data[dvar])
        else:
            cols = [f"{dvar}_X", f"{dvar}_Y", f"{dvar}_Z"]
            df[cols] = data[dvar].values
            metavar_index += 1

    df.set_index("datetime", inplace=True)
    return df


def read_data(path, pattern=None, pattern_to_read=["csv"]):
    """
    Reads data from the files downloaded using the download_data module.
    """
    dfs = []
    files = natsort.natsorted(os.listdir(path))
    for f in files:
        skip = False
        if pattern is not None:
            if pattern not in f:
                continue
        if pattern_to_read is not None:
            for pat in pattern_to_read:
                if f.find(pat) < 0:
                    skip = True
        if skip:
            continue
        file_path = os.path.join(path, f)
        df = pd.read_csv(file_path, parse_dates=["datetime"], index_col="datetime")
        df.sort_index(inplace=True)
        dfs.append(df)
    return dfs

