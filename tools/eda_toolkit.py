from __future__ import annotations

import warnings
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union, Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# Silence common non-actionable warnings for EDA notebooks without hiding real errors.
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


class DataInquisitor:
    """
    Automated Exploratory Data Analysis (EDA) toolkit for pandas DataFrames.

    This class is designed to be imported into Jupyter notebooks and used as a reusable
    EDA helper. It includes defensive programming guardrails to handle common issues
    (empty dataframes, non-numeric columns, zero variance columns, inf values, memory-heavy
    plots) without crashing the notebook.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset.
    target : str, optional
        Optional target column name used for coloring (hue) in certain plots (e.g., project_2d).
    config : Mapping[str, Any], optional
        Configuration dictionary for aesthetics and runtime behavior. Supported keys:
        - "figsize": Tuple[float, float] figure size in inches (default: (12, 6))
        - "style": seaborn style (default: "whitegrid")
        - "context": seaborn context (default: "notebook")
        - "palette": seaborn palette name (default: "deep")
        - "dpi": integer DPI for figures (default: 110)
        - "random_state": integer random seed for sampling and TSNE (default: 42)
    copy : bool, default True
        If True, store a copy of the DataFrame to avoid side-effects on the caller's object.

    Attributes
    ----------
    df : pandas.DataFrame
        Internal DataFrame (copied unless copy=False).
    target : str or None
        Target column name used for hue in projections/plots when applicable.
    config : Dict[str, Any]
        Resolved configuration dictionary.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target: Optional[str] = None,
        config: Optional[Mapping[str, Any]] = None,
        copy: bool = True,
    ) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"df must be a pandas.DataFrame, got {type(df)}")

        self.df: pd.DataFrame = df.copy(deep=True) if copy else df
        self.target: Optional[str] = target

        default_config: Dict[str, Any] = {
            "figsize": (12.0, 6.0),
            "style": "whitegrid",
            "context": "notebook",
            "palette": "deep",
            "dpi": 110,
            "random_state": 42,
        }
        user_config: Dict[str, Any] = dict(config) if config is not None else {}
        self.config: Dict[str, Any] = {**default_config, **user_config}

        # Configure seaborn aesthetics globally (typical notebook usage).
        try:
            sns.set_theme(
                style=str(self.config.get("style", "whitegrid")),
                context=str(self.config.get("context", "notebook")),
                palette=str(self.config.get("palette", "deep")),
            )
        except Exception as exc:
            # Never fail initialization due to theme settings.
            print(
                f"[DataInquisitor] Warning: seaborn theme configuration failed: {exc}"
            )

        try:
            plt.rcParams["figure.dpi"] = int(self.config.get("dpi", 110))
        except Exception:
            # If matplotlib rcParams fails for any reason, ignore.
            pass

    # --------------------------
    # Public utilities
    # --------------------------

    def health_check(self) -> Dict[str, Any]:
        """
        Print high-level dataset diagnostics and visualize missing values if present.

        The method prints:
        - DataFrame shape
        - Duplicate row count
        - Column dtypes
        - Missing value counts summary

        It also plots a bar chart of missing values per column (only if any missing exist).

        Returns
        -------
        Dict[str, Any]
            Summary dictionary with keys:
            - "shape": Tuple[int, int]
            - "duplicate_rows": int
            - "dtypes": pandas.Series of dtypes
            - "missing_by_column": pandas.Series of missing counts (sorted desc)
            - "missing_total": int
        """
        if self._is_empty_df():
            print("[DataInquisitor] health_check: DataFrame is empty.")
            return {
                "shape": (0, 0),
                "duplicate_rows": 0,
                "dtypes": pd.Series(dtype="object"),
                "missing_by_column": pd.Series(dtype="int64"),
                "missing_total": 0,
            }

        shape = self.df.shape
        duplicate_rows = int(self.df.duplicated().sum())
        dtypes = self.df.dtypes

        missing_by_column = self.df.isna().sum().sort_values(ascending=False)
        missing_total = int(missing_by_column.sum())

        print(f"[DataInquisitor] Shape: {shape[0]} rows x {shape[1]} columns")
        print(f"[DataInquisitor] Duplicate rows: {duplicate_rows}")
        print("[DataInquisitor] Dtypes:")
        print(dtypes.to_string())

        if missing_total > 0:
            missing_nonzero = missing_by_column[missing_by_column > 0]
            print(f"[DataInquisitor] Missing values total: {missing_total}")
            print("[DataInquisitor] Missing values by column (top 20):")
            print(missing_nonzero.head(20).to_string())

            try:
                fig, ax = plt.subplots(figsize=self._figsize())
                sns.barplot(
                    x=missing_nonzero.index.astype(str),
                    y=missing_nonzero.values.astype(float),
                    ax=ax,
                )
                ax.set_title("Missing Values by Column")
                ax.set_xlabel("Column")
                ax.set_ylabel("Missing count")
                ax.tick_params(axis="x", rotation=60, labelsize=9)
                plt.tight_layout()
                plt.show()
                plt.close(fig)
            except Exception as exc:
                print(f"[DataInquisitor] health_check: missing plot failed: {exc}")
        else:
            print("[DataInquisitor] Missing values total: 0 (no missingness detected)")

        return {
            "shape": shape,
            "duplicate_rows": duplicate_rows,
            "dtypes": dtypes,
            "missing_by_column": missing_by_column,
            "missing_total": missing_total,
        }

    def clean_names(self) -> "DataInquisitor":
        """
        Clean column names by lowercasing and normalizing whitespace.

        Operations:
        - Strip leading/trailing whitespace
        - Convert to lowercase
        - Replace internal whitespace with underscores
        - Ensure uniqueness by adding suffixes if collisions occur

        Returns
        -------
        DataInquisitor
            Self (enables chaining).
        """
        if self._is_empty_df():
            return self

        original_cols = list(self.df.columns)
        cleaned = (
            pd.Index(original_cols)
            .astype(str)
            .str.strip()
            .str.lower()
            .str.replace(r"\s+", "_", regex=True)
        )

        # Ensure uniqueness after cleaning.
        if cleaned.has_duplicates:
            cleaned = self._make_unique_columns(cleaned)

        self.df.columns = cleaned
        return self

    def plot_distributions(
        self, columns: Optional[Sequence[str]] = None, top_n: int = 12
    ) -> None:
        """
        Plot distributions for numeric columns with a boxplot and histogram+KDE+normal overlay.

        For each selected numeric column:
        - Top subplot: horizontal boxplot (outliers visible)
        - Bottom subplot: histogram (density) + KDE + fitted normal PDF overlay

        Notes:
        - Automatically ignores non-numeric columns.
        - Replaces +/-inf with NaN before plotting.
        - Skips columns with no usable (non-NaN) values.
        - Skips columns with zero variance (std == 0).

        Parameters
        ----------
        columns : Sequence[str], optional
            Columns to consider. If None, all numeric columns are considered.
        top_n : int, default 12
            Maximum number of numeric columns to plot. If more are available, the method
            selects the `top_n` columns by variance (descending) for more informative plots.

        Returns
        -------
        None
        """
        if self._is_empty_df():
            print("[DataInquisitor] plot_distributions: DataFrame is empty.")
            return

        numeric_cols = self._get_numeric_columns(columns=columns, include_bool=False)

        if not numeric_cols:
            print("[DataInquisitor] plot_distributions: No numeric columns to plot.")
            return

        numeric_cols = self._select_top_by_variance(numeric_cols, top_n=top_n)

        for col in numeric_cols:
            try:
                s = self._numeric_series(col)
                if s.empty:
                    print(
                        f"[DataInquisitor] plot_distributions: '{col}' has no usable values."
                    )
                    continue

                std = float(s.std(ddof=0))
                if not np.isfinite(std) or std <= 0.0:
                    print(
                        f"[DataInquisitor] plot_distributions: '{col}' has zero variance; skipping."
                    )
                    continue

                mean = float(s.mean())
                x_min, x_max = float(s.min()), float(s.max())

                # Robust histogram range to reduce the visual dominance of extreme outliers.
                lo, hi = self._robust_range(s)
                hist_range = (
                    (lo, hi)
                    if np.isfinite(lo) and np.isfinite(hi) and lo < hi
                    else (x_min, x_max)
                )

                fig, axes = plt.subplots(
                    2, 1, figsize=self._figsize(height_multiplier=1.2), sharex=True
                )

                # Boxplot (outliers visible).
                sns.boxplot(x=s.values, ax=axes[0], orient="h")
                axes[0].set_title(f"Boxplot: {col}")
                axes[0].set_xlabel("")

                # Histogram + KDE.
                sns.histplot(
                    s.values,
                    ax=axes[1],
                    kde=True,
                    stat="density",
                    bins="auto",
                    binrange=hist_range,
                )

                # Normal overlay.
                x = np.linspace(hist_range[0], hist_range[1], 300)
                pdf = stats.norm.pdf(x, loc=mean, scale=std)
                axes[1].plot(x, pdf, linewidth=2, label="Normal fit (PDF)")
                axes[1].set_title(f"Histogram + KDE + Normal Fit: {col}")
                axes[1].set_xlabel(col)
                axes[1].set_ylabel("Density")
                axes[1].legend(loc="best")

                plt.tight_layout()
                plt.show()
                plt.close(fig)

            except Exception as exc:
                print(f"[DataInquisitor] plot_distributions: failed for '{col}': {exc}")

    def plot_categorical(self, top_n: int = 15) -> None:
        """
        Plot categorical feature distributions as bar charts with percentage annotations.

        Guardrails:
        - Skips features with > 50 unique categories (high cardinality).
        - If categories exceed `top_n`, only the top categories are shown and the rest are
        aggregated as "Other".
        - Handles NaN as an explicit category label.

        Parameters
        ----------
        top_n : int, default 15
            Maximum number of categories to display per feature.

        Returns
        -------
        None
        """
        if self._is_empty_df():
            print("[DataInquisitor] plot_categorical: DataFrame is empty.")
            return

        cat_cols = self._get_categorical_columns()
        if not cat_cols:
            print("[DataInquisitor] plot_categorical: No categorical columns to plot.")
            return

        for col in cat_cols:
            try:
                vc = (
                    self.df[col]
                    .astype("object")
                    .where(self.df[col].notna(), other="NaN")
                    .value_counts(dropna=False)
                )
                unique_n = int(vc.shape[0])

                if unique_n == 0:
                    continue

                if unique_n > 50:
                    print(
                        f"[DataInquisitor] plot_categorical: '{col}' has {unique_n} categories (>50); skipping."
                    )
                    continue

                total = float(vc.sum())
                if total <= 0:
                    continue

                if unique_n > top_n:
                    top = vc.head(top_n)
                    other_sum = float(vc.iloc[top_n:].sum())
                    vc_plot = top.copy()
                    if other_sum > 0:
                        vc_plot.loc["Other"] = other_sum
                else:
                    vc_plot = vc

                labels = vc_plot.index.astype(str)
                counts = vc_plot.values.astype(float)
                perc = counts / total * 100.0

                fig, ax = plt.subplots(figsize=self._figsize(height_multiplier=1.1))
                sns.barplot(x=labels, y=counts, ax=ax)
                ax.set_title(f"Categorical Distribution: {col}")
                ax.set_xlabel(col)
                ax.set_ylabel("Count")
                ax.tick_params(axis="x", rotation=45, labelsize=9)

                # Percentage annotations.
                for i, (c, p) in enumerate(zip(counts, perc)):
                    if np.isfinite(c) and c > 0:
                        ax.text(i, c, f"{p:.1f}%", ha="center", va="bottom", fontsize=9)

                plt.tight_layout()
                plt.show()
                plt.close(fig)

            except Exception as exc:
                print(f"[DataInquisitor] plot_categorical: failed for '{col}': {exc}")

    def plot_correlations(self, threshold: float = 0.3) -> None:
        """
        Plot a correlation heatmap for numeric features, filtering weak correlations.

        The method:
        - Selects numeric columns only (non-numeric columns are ignored).
        - Computes Pearson correlation matrix.
        - Filters correlations with abs(corr) < threshold by setting them to NaN.
        - Drops columns/rows that have no remaining strong correlations (excluding diagonal).

        Parameters
        ----------
        threshold : float, default 0.3
            Minimum absolute correlation to display. Values below this are filtered out.

        Returns
        -------
        None
        """
        if self._is_empty_df():
            print("[DataInquisitor] plot_correlations: DataFrame is empty.")
            return

        try:
            threshold_val = float(threshold)
        except Exception:
            print(
                "[DataInquisitor] plot_correlations: threshold must be a float in [0, 1]."
            )
            return

        if not (0.0 <= threshold_val <= 1.0):
            print("[DataInquisitor] plot_correlations: threshold must be in [0, 1].")
            return

        numeric_cols = self._get_numeric_columns(include_bool=False)
        if len(numeric_cols) < 2:
            print(
                "[DataInquisitor] plot_correlations: Need at least 2 numeric columns."
            )
            return

        try:
            num_df = self.df[numeric_cols].replace([np.inf, -np.inf], np.nan)
            corr = num_df.corr(method="pearson")

            # Replace NaN correlations from constant columns with 0 so thresholding can remove them.
            corr = corr.fillna(0.0)

            # Identify strong correlations excluding the diagonal.
            n = corr.shape[0]
            diag = np.eye(n, dtype=bool)
            strong_mask = (corr.abs().values >= threshold_val) & (~diag)

            cols_with_strong = strong_mask.any(axis=0)
            if not np.any(cols_with_strong):
                print(
                    f"[DataInquisitor] plot_correlations: No correlations >= {threshold_val:.2f}."
                )
                return

            keep_cols = list(corr.columns[cols_with_strong])
            corr_small = corr.loc[keep_cols, keep_cols].copy()

            # Filter weak correlations (< threshold) by setting them to NaN (noise reduction).
            corr_filtered = corr_small.where(
                corr_small.abs() >= threshold_val, other=np.nan
            )

            # Keep diagonal for context.
            for i in range(corr_filtered.shape[0]):
                corr_filtered.iat[i, i] = 1.0

            # Plot lower triangle to reduce visual redundancy.
            mask_upper = np.triu(np.ones_like(corr_filtered, dtype=bool), k=1)

            fig, ax = plt.subplots(
                figsize=self._figsize(width_multiplier=1.2, height_multiplier=1.1)
            )
            sns.heatmap(
                corr_filtered,
                mask=mask_upper,
                ax=ax,
                center=0.0,
                cmap="vlag",
                linewidths=0.5,
                linecolor="white",
                cbar_kws={"shrink": 0.8},
            )
            ax.set_title(f"Correlation Heatmap (|corr| >= {threshold_val:.2f})")
            plt.tight_layout()
            plt.show()
            plt.close(fig)

        except Exception as exc:
            print(f"[DataInquisitor] plot_correlations: failed: {exc}")

    def plot_pairplot(self, sample_size: int = 1000) -> None:
        """
        Plot a seaborn pairplot with safe downsampling to avoid memory issues.

        Guardrails:
        - Downsamples to at most min(sample_size, 1000, len(df)) rows.
        - Uses only numeric columns.
        - If many numeric columns exist, selects up to 10 columns by variance.
        - Drops rows with NaN/Inf in selected columns.
        - Uses hue=self.target only if the target exists and has manageable cardinality.

        Parameters
        ----------
        sample_size : int, default 1000
            Requested maximum number of rows for the pairplot. The method hard-caps at 1000.

        Returns
        -------
        None
        """
        if self._is_empty_df():
            print("[DataInquisitor] plot_pairplot: DataFrame is empty.")
            return

        numeric_cols = self._get_numeric_columns(include_bool=False)
        numeric_cols = self._drop_zero_variance_columns(numeric_cols)

        if len(numeric_cols) < 2:
            print(
                "[DataInquisitor] plot_pairplot: Need at least 2 numeric columns after filtering."
            )
            return

        # Hard guardrail against huge pairplots.
        if len(numeric_cols) > 10:
            numeric_cols = self._select_top_by_variance(numeric_cols, top_n=10)

        max_rows = min(int(sample_size), 1000, int(len(self.df)))
        if max_rows <= 1:
            print("[DataInquisitor] plot_pairplot: Not enough rows to plot.")
            return

        use_cols = list(numeric_cols)
        hue_col: Optional[str] = None
        if self.target and self.target in self.df.columns:
            hue_col = self.target
            if hue_col not in use_cols:
                use_cols_with_hue = use_cols + [hue_col]
            else:
                use_cols_with_hue = use_cols
        else:
            use_cols_with_hue = use_cols

        try:
            data = (
                self.df[use_cols_with_hue]
                .replace([np.inf, -np.inf], np.nan)
                .dropna(axis=0, how="any")
            )
            if data.empty or len(data) < 2:
                print(
                    "[DataInquisitor] plot_pairplot: No usable rows after dropping NaN/Inf."
                )
                return

            if len(data) > max_rows:
                data = data.sample(
                    n=max_rows, random_state=int(self.config.get("random_state", 42))
                )

            # Hue guardrail (avoid massive legends).
            hue_to_use: Optional[str] = None
            if hue_col and hue_col in data.columns:
                nunique = int(data[hue_col].nunique(dropna=False))
                if nunique <= 20:
                    hue_to_use = hue_col
                else:
                    print(
                        f"[DataInquisitor] plot_pairplot: target '{hue_col}' has {nunique} unique values; "
                        "not using hue to avoid clutter."
                    )

            grid = sns.pairplot(
                data=data,
                vars=use_cols,
                hue=hue_to_use,
                diag_kind="hist",
                corner=True,
                plot_kws={"s": 18, "alpha": 0.75},
            )
            grid.fig.suptitle("Pairplot (downsampled)", y=1.02)
            plt.tight_layout()
            plt.show()
            plt.close(grid.fig)

        except Exception as exc:
            print(f"[DataInquisitor] plot_pairplot: failed: {exc}")

    def project_2d(self, method: Literal["pca", "tsne"] = "pca") -> pd.DataFrame:
        """
        Standardize numeric data, remove NaNs/Inf, project to 2D, and scatter plot (with hue if target is set).

        Pipeline:
        1) Select numeric features (excluding the target column if present).
        2) Replace +/-inf with NaN and drop rows with missing values in features.
        3) Drop zero-variance features.
        4) Standardize features (StandardScaler).
        5) Project to 2D using PCA or t-SNE.
        6) Scatter plot. If target exists, color points by target.

        Robustness:
        - Handles empty DataFrames and insufficient numeric features.
        - Adjusts t-SNE perplexity based on sample size.
        - Catches projection errors (e.g., too few samples, numerical issues).

        Parameters
        ----------
        method : {'pca', 'tsne'}, default 'pca'
            Projection method to use.

        Returns
        -------
        pandas.DataFrame
            DataFrame with columns:
            - "x": float, first component
            - "y": float, second component
            - target column (if available): original target values aligned to plotted points
        """
        if self._is_empty_df():
            print("[DataInquisitor] project_2d: DataFrame is empty.")
            return pd.DataFrame(columns=["x", "y"])

        method_lc = str(method).strip().lower()
        if method_lc not in {"pca", "tsne"}:
            raise ValueError("method must be 'pca' or 'tsne'")

        numeric_cols = self._get_numeric_columns(include_bool=False)
        if self.target and self.target in numeric_cols:
            numeric_cols = [c for c in numeric_cols if c != self.target]

        numeric_cols = self._drop_zero_variance_columns(numeric_cols)
        if len(numeric_cols) < 2:
            print(
                "[DataInquisitor] project_2d: Need at least 2 numeric features after filtering."
            )
            return pd.DataFrame(columns=["x", "y"])

        try:
            X_raw = self.df[numeric_cols].replace([np.inf, -np.inf], np.nan)
            valid_mask = X_raw.notna().all(axis=1)
            X_clean = X_raw.loc[valid_mask].astype(float)

            if X_clean.empty or len(X_clean) < 3:
                print(
                    "[DataInquisitor] project_2d: Not enough clean rows after dropping NaN/Inf."
                )
                return pd.DataFrame(columns=["x", "y"])

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_clean.values)

            if not np.isfinite(X_scaled).all():
                # Extremely defensive: in case scaling produced non-finite values.
                finite_rows = np.isfinite(X_scaled).all(axis=1)
                X_scaled = X_scaled[finite_rows]
                X_clean = X_clean.iloc[finite_rows]
                if len(X_clean) < 3:
                    print(
                        "[DataInquisitor] project_2d: Not enough finite rows after scaling."
                    )
                    return pd.DataFrame(columns=["x", "y"])

            if method_lc == "pca":
                projector = PCA(n_components=2)
                Z = projector.fit_transform(X_scaled)
                title = "2D Projection (PCA)"
            else:
                # t-SNE: adjust perplexity to sample size constraints.
                n_samples = int(X_scaled.shape[0])
                # Perplexity must be < n_samples; typical default is 30.
                # Use a conservative heuristic and clamp to [5, 30] when possible.
                max_perplex = max(5, min(30, (n_samples - 1) // 3))
                perplexity = float(min(30, max_perplex))
                if n_samples <= 10:
                    perplexity = float(max(2, min(5, n_samples - 1)))

                projector = TSNE(
                    n_components=2,
                    random_state=int(self.config.get("random_state", 42)),
                    perplexity=perplexity,
                    init="pca",
                    learning_rate="auto",
                )
                Z = projector.fit_transform(X_scaled)
                title = f"2D Projection (t-SNE, perplexity={perplexity:g})"

            proj_df = pd.DataFrame(Z, columns=["x", "y"], index=X_clean.index)

            hue_values: Optional[pd.Series] = None
            if self.target and self.target in self.df.columns:
                hue_values = self.df.loc[proj_df.index, self.target]
                proj_df[self.target] = hue_values.values

            # Plot
            try:
                fig, ax = plt.subplots(
                    figsize=self._figsize(width_multiplier=1.1, height_multiplier=1.1)
                )

                if hue_values is None:
                    ax.scatter(
                        proj_df["x"].values, proj_df["y"].values, s=22, alpha=0.8
                    )
                else:
                    # Numeric hue -> use colorbar; categorical hue -> seaborn with legend
                    if (
                        pd.api.types.is_numeric_dtype(hue_values)
                        and int(pd.Series(hue_values).nunique(dropna=False)) > 20
                    ):
                        sc = ax.scatter(
                            proj_df["x"].values,
                            proj_df["y"].values,
                            c=pd.to_numeric(hue_values, errors="coerce").values,
                            s=22,
                            alpha=0.85,
                        )
                        cbar = plt.colorbar(sc, ax=ax)
                        cbar.set_label(str(self.target))
                    else:
                        plot_df = proj_df.copy()
                        plot_df[self.target] = (
                            plot_df[self.target]
                            .astype("object")
                            .where(
                                pd.notna(plot_df[self.target]),
                                other="NaN",
                            )
                        )
                        sns.scatterplot(
                            data=plot_df,
                            x="x",
                            y="y",
                            hue=self.target,
                            ax=ax,
                            s=35,
                            alpha=0.85,
                            linewidth=0.0,
                        )
                        ax.legend(
                            title=str(self.target),
                            bbox_to_anchor=(1.02, 1),
                            loc="upper left",
                            borderaxespad=0,
                        )

                ax.set_title(title)
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                plt.tight_layout()
                plt.show()
                plt.close(fig)
            except Exception as exc:
                print(f"[DataInquisitor] project_2d: plot failed: {exc}")

            return proj_df.reset_index(drop=True)

        except Exception as exc:
            print(f"[DataInquisitor] project_2d: projection failed: {exc}")
            return pd.DataFrame(columns=["x", "y"])

    # --------------------------
    # Internal helpers
    # --------------------------

    def _is_empty_df(self) -> bool:
        """
        Check if internal DataFrame is empty or missing.

        Returns
        -------
        bool
            True if DataFrame is empty or has no columns, else False.
        """
        return (
            self.df is None
            or not isinstance(self.df, pd.DataFrame)
            or self.df.empty
            or self.df.shape[1] == 0
        )

    def _figsize(
        self,
        width_multiplier: float = 1.0,
        height_multiplier: float = 1.0,
        height_multiplier_if_tall: float = 1.0,
    ) -> Tuple[float, float]:
        """
        Compute figure size based on config and optional multipliers.

        Parameters
        ----------
        width_multiplier : float, default 1.0
            Multiplier for figure width.
        height_multiplier : float, default 1.0
            Multiplier for figure height.
        height_multiplier_if_tall : float, default 1.0
            Deprecated placeholder to avoid breaking older notebooks; not used.

        Returns
        -------
        Tuple[float, float]
            (width, height) in inches.
        """
        base = self.config.get("figsize", (12.0, 6.0))
        try:
            w, h = float(base[0]), float(base[1])
        except Exception:
            w, h = 12.0, 6.0
        return (
            max(4.0, w * float(width_multiplier)),
            max(3.0, h * float(height_multiplier)),
        )

    def _get_numeric_columns(
        self,
        columns: Optional[Sequence[str]] = None,
        include_bool: bool = False,
    ) -> List[str]:
        """
        Get numeric columns from the DataFrame.

        Parameters
        ----------
        columns : Sequence[str], optional
            Candidate columns. If None, consider all columns.
        include_bool : bool, default False
            If True, boolean columns are included as numeric.

        Returns
        -------
        List[str]
            List of numeric column names present in the DataFrame.
        """
        if self._is_empty_df():
            return []

        if columns is None:
            cols = list(self.df.columns)
        else:
            cols = [c for c in columns if c in self.df.columns]

        if not cols:
            return []

        if include_bool:
            numeric_df = self.df[cols].select_dtypes(include=[np.number, "bool"])
        else:
            numeric_df = (
                self.df[cols]
                .select_dtypes(include=[np.number])
                .select_dtypes(exclude=["bool"])
            )

        return list(numeric_df.columns)

    def _get_categorical_columns(self) -> List[str]:
        """
        Get categorical columns from the DataFrame.

        Returns
        -------
        List[str]
            List of categorical column names (object/category/string/bool).
        """
        if self._is_empty_df():
            return []

        cat_df = self.df.select_dtypes(include=["object", "category", "string", "bool"])
        return list(cat_df.columns)

    def _numeric_series(self, col: str) -> pd.Series:
        """
        Return a sanitized numeric series for plotting.

        - Converts to numeric (coerce errors).
        - Replaces +/-inf with NaN.
        - Drops NaN.

        Parameters
        ----------
        col : str
            Column name.

        Returns
        -------
        pandas.Series
            Cleaned numeric series.
        """
        s = pd.to_numeric(self.df[col], errors="coerce")
        s = s.replace([np.inf, -np.inf], np.nan).dropna()
        return s

    def _drop_zero_variance_columns(self, cols: Sequence[str]) -> List[str]:
        """
        Drop columns with zero variance (or non-finite variance).

        Parameters
        ----------
        cols : Sequence[str]
            Numeric columns.

        Returns
        -------
        List[str]
            Filtered list with non-constant columns.
        """
        if self._is_empty_df() or not cols:
            return []

        kept: List[str] = []
        for c in cols:
            try:
                s = self._numeric_series(c)
                if s.empty:
                    continue
                v = float(s.var(ddof=0))
                if np.isfinite(v) and v > 0.0:
                    kept.append(c)
            except Exception:
                continue
        return kept

    def _select_top_by_variance(self, cols: Sequence[str], top_n: int) -> List[str]:
        """
        Select top columns by variance (descending).

        Parameters
        ----------
        cols : Sequence[str]
            Candidate numeric columns.
        top_n : int
            Maximum number of columns to keep.

        Returns
        -------
        List[str]
            Selected column names.
        """
        if self._is_empty_df() or not cols:
            return []

        n = max(1, int(top_n))
        if len(cols) <= n:
            return list(cols)

        vars_list: List[Tuple[str, float]] = []
        for c in cols:
            try:
                s = self._numeric_series(c)
                if s.empty:
                    continue
                v = float(s.var(ddof=0))
                if np.isfinite(v):
                    vars_list.append((c, v))
            except Exception:
                continue

        if not vars_list:
            return list(cols)[:n]

        vars_list.sort(key=lambda x: x[1], reverse=True)
        return [c for c, _ in vars_list[:n]]

    def _robust_range(self, s: pd.Series) -> Tuple[float, float]:
        """
        Compute a robust plotting range using percentiles to mitigate extreme outliers.

        Parameters
        ----------
        s : pandas.Series
            Numeric series.

        Returns
        -------
        Tuple[float, float]
            (low, high) based on 1st and 99th percentile, with fallback to min/max if needed.
        """
        try:
            vals = s.values.astype(float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                return (float("nan"), float("nan"))
            lo = float(np.nanpercentile(vals, 1))
            hi = float(np.nanpercentile(vals, 99))
            if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
                return (float(np.nanmin(vals)), float(np.nanmax(vals)))
            return (lo, hi)
        except Exception:
            return (float("nan"), float("nan"))

    def _make_unique_columns(self, cols: pd.Index) -> pd.Index:
        """
        Make column names unique by appending a numeric suffix.

        Parameters
        ----------
        cols : pandas.Index
            Column names.

        Returns
        -------
        pandas.Index
            Unique column names.
        """
        seen: Dict[str, int] = {}
        unique: List[str] = []
        for c in cols.astype(str).tolist():
            if c not in seen:
                seen[c] = 0
                unique.append(c)
            else:
                seen[c] += 1
                unique.append(f"{c}__{seen[c]}")
        return pd.Index(unique)


if __name__ == "__main__":
    # Dummy usage example (intended for quick smoke testing).
    # Note: sns.load_dataset may require internet depending on environment.
    df_demo = sns.load_dataset("titanic")
    inquisitor = DataInquisitor(
        df_demo, target="survived", config={"figsize": (12, 6), "random_state": 42}
    )
    inquisitor.clean_names()
    inquisitor.health_check()
    inquisitor.plot_distributions(top_n=6)
    inquisitor.plot_categorical(top_n=12)
    inquisitor.plot_correlations(threshold=0.3)
    inquisitor.plot_pairplot(sample_size=600)
    inquisitor.project_2d(method="pca")
