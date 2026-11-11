# STAGE 1: load raw data, visualize it, clean it, save a preprocessed CSV.
# Reusable: change the settings below to use it with any CSV.
import pandas as pd
import matplotlib.pyplot as plt

INPUT_PATH = "data/raw/Mall_Customers.csv"
OUTPUT_PATH = "data/processed/mall_preprocessed.csv"
PLOT_FOLDER = "data/processed/"                  # where the charts are saved (folder must exist)
DROP_COLUMNS = ["CustomerID", "Gender"]          # columns not used for clustering
RENAME_COLUMNS = {"Annual Income (k$)": "Income", "Spending Score (1-100)": "SpendingScore"}
SHOW_PLOTS = False                               # True = also open the chart windows


def load_data(path):
    return pd.read_csv(path)


def preprocess(df, drop_columns, rename_columns):
    df = df.copy()
    df = df.drop(columns=[c for c in drop_columns if c in df.columns])
    df = df.rename(columns=rename_columns)
    df = df.drop_duplicates()

    # Missing values: median for numeric columns, mode for text columns
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])
    return df


# ---------------- Data visualization (matplotlib only) ----------------

def finish_plot(file_name):
    plt.tight_layout()
    plt.savefig(PLOT_FOLDER + file_name, dpi=110)
    if SHOW_PLOTS:
        plt.show()
    plt.close()


def plot_categorical_counts(df):
    # Bar chart for every text column (e.g. Gender). Use on raw data, before dropping.
    cat_cols = df.select_dtypes(exclude="number").columns
    for col in cat_cols:
        if df[col].nunique() > 15:               # skip ID-like columns
            continue
        counts = df[col].value_counts()
        plt.figure(figsize=(5, 4))
        bars = plt.bar(counts.index.astype(str), counts.values, color="steelblue")
        plt.bar_label(bars)
        plt.title("Count of " + col)
        plt.ylabel("Customers")
        finish_plot("plot_count_" + col + ".png")


def plot_histograms(df):
    num_cols = df.columns
    fig, axes = plt.subplots(1, len(num_cols), figsize=(5 * len(num_cols), 4))
    for ax, col in zip(axes, num_cols):
        ax.hist(df[col], bins=20, color="steelblue", edgecolor="white")
        ax.axvline(df[col].mean(), color="red", linestyle="--", label="mean")
        ax.set_title("Distribution of " + col)
        ax.set_xlabel(col)
        ax.legend()
    finish_plot("plot_histograms.png")


def plot_boxplots(df):
    # Boxplots show outliers (dots beyond the whiskers)
    num_cols = df.columns
    fig, axes = plt.subplots(1, len(num_cols), figsize=(4 * len(num_cols), 4))
    for ax, col in zip(axes, num_cols):
        ax.boxplot(df[col], patch_artist=True)
        ax.set_title("Boxplot of " + col)
    finish_plot("plot_boxplots.png")


def plot_correlation(df):
    corr = df.corr()
    plt.figure(figsize=(5.5, 5))
    plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar()
    plt.xticks(range(len(corr)), corr.columns, rotation=45)
    plt.yticks(range(len(corr)), corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            plt.text(j, i, round(corr.iloc[i, j], 2), ha="center", va="center")
    plt.title("Correlation Heatmap")
    finish_plot("plot_correlation.png")


def plot_scatter_pairs(df):
    # One scatter plot for every pair of columns (helps to spot natural groups)
    cols = list(df.columns)
    pairs = [(cols[i], cols[j]) for i in range(len(cols)) for j in range(i + 1, len(cols))]
    fig, axes = plt.subplots(1, len(pairs), figsize=(5 * len(pairs), 4))
    for ax, (x, y) in zip(axes, pairs):
        ax.scatter(df[x], df[y], s=20, alpha=0.7, color="darkorange")
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(x + " vs " + y)
    finish_plot("plot_scatter_pairs.png")


if __name__ == "__main__":
    raw = load_data(INPUT_PATH)
    print("Raw shape:", raw.shape)
    print("Missing values:\n", raw.isnull().sum())

    plot_categorical_counts(raw)                 # raw data, so Gender is still there

    clean = preprocess(raw, DROP_COLUMNS, RENAME_COLUMNS)
    print("Clean shape:", clean.shape)
    print(clean.describe())

    plot_histograms(clean)
    plot_boxplots(clean)
    plot_correlation(clean)
    plot_scatter_pairs(clean)
    print("Charts saved in:", PLOT_FOLDER)

    clean.to_csv(OUTPUT_PATH, index=False)
    print("Saved:", OUTPUT_PATH)