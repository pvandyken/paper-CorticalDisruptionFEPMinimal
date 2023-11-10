from __future__ import annotations
import itertools as it
from typing import Any, Callable, DefaultDict, Hashable, Literal, TypeAlias
import warnings

import more_itertools as itx
import numpy as np
import pandas as pd
import scipy.stats as scs

ReportType: TypeAlias = Literal["mean", "mode", "all"]


class SummaryScale:
    def __init__(self, data: pd.Series[Any]):
        self.field = data.name
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            self.mean = data.mean()
            self.std = data.std()
            self.median = data.median()
            self.iqr = scs.iqr(data, nan_policy="omit")

    def __repr__(self):
        return f"{self.field}: {self.format_all()}"

    def format_all(self):
        if np.isnan(self.mean):
            return "N/A"
        return f"{self.mean:.2f} ({self.median:.2f}) ± {self.std:.2f} ({self.iqr:.2f})"

    def format_mean(self):
        if np.isnan(self.mean):
            return "N/A"
        return f"{self.mean:.2f} ({self.std:.2f})"

    def format_median(self):
        if np.isnan(self.mean):
            return "N/A"
        return f"{self.median:.2f} ({self.iqr:.2f})"


class SummaryNominal:
    def __init__(self, data: pd.Series[Any], template: str | None = None):
        self.template = template
        self.field = data.name
        self.table = DefaultDict(lambda: 0, data.value_counts().items())

    def __repr__(self):
        formatted = self.format()
        if self.template is None:
            return f"{self.field}: {formatted}"
        return f"{self.field} ({self.format_template()}): {formatted}"

    def format_template(self):
        if self.template is not None:
            return self.template.replace("{", "").replace("}", "")
        return ""

    def format(self):
        if self.template is None:
            return repr(self.table)
        return self.template.format_map(self.table)


class DemographicTable:
    def __init__(self, data: pd.DataFrame, groups: str, order=None):
        self.grouped = data.groupby(groups)
        self.labels = list(order or self.grouped.groups)
        self.n = {label: len(vals) for label, vals in self.grouped.groups.items()}
        self.table: dict[Hashable, dict[str, str]] = {
            label: {} for label in self.labels
        }
        self.interaction_labels = {
            labels: self._format_interaction(labels)
            for labels in it.combinations(self.labels, 2)
        }
        self.interactions = {
            interaction: {} for interaction in self.interaction_labels.values()
        }

    @staticmethod
    def _format_interaction(interaction: tuple[Hashable, Hashable]):
        return f"{interaction[0]} vs {interaction[1]}"

    @staticmethod
    def _format_pvalue(pvalue: float, alpha: float = 0.05):
        if pvalue < 0.001:
            formatted = f"_p_ < 0.001"
        else:
            formatted = f"_p_ = {pvalue:.2g}"
        if pvalue <= alpha:
            return f"**{formatted}**"
        return formatted

    @staticmethod
    def _format_chi(dof: int, chi: float):
        return f"χ\u00b2({dof}) = {chi:.3g}"

    @staticmethod
    def _format_t(dof: int, t: float):
        return f"_t_({dof}) = {t:.3g}"

    @staticmethod
    def _format_field(field: str, key: str | None = None):
        if key is not None:
            return f"{field} ({key})"
        return field

    @staticmethod
    def _format_scale(scale: SummaryScale, report: ReportType):
        if report == "mean":
            return scale.format_mean()
        if report == "median":
            return scale.format_median()
        if report == "all":
            return scale.format_all()
        raise ValueError(f"Unrecognized report: '{report}'")

    def add_nominal(
        self,
        field: str,
        template: str | None = None,
        name: str | None = None,
        autoformatter: Callable[[str], str] = lambda i: i,
    ):
        summaries = {
            label: SummaryNominal(self.grouped.get_group(label)[field], template)
            for label in self.labels
        }
        name = self._format_field(
            name or autoformatter(field),
            itx.first(summaries.values()).format_template(),
        )
        for label in self.labels:
            self.table[label][name] = summaries[label].format()
        for vars, label in self.interaction_labels.items():
            contigency = pd.DataFrame({var: summaries[var].table for var in vars})
            statistic, pvalue, dof, _ = scs.chi2_contingency(contigency)
            self.interactions[label][name] = ", ".join(
                [self._format_chi(dof, statistic), self._format_pvalue(pvalue)]
            )

    def add_scale(
        self,
        field: str,
        name: str | None = None,
        autoformatter: Callable[[str], str] = lambda i: i,
        report: ReportType = "mean",
        skip_stats: bool = False,
    ):
        name = self._format_field(
            name or autoformatter(field),
        )
        if report == "median":
            name += " - median (IQR)"
        for label in self.labels:
            self.table[label][name] = self._format_scale(
                SummaryScale(self.grouped.get_group(label)[field]), report
            )

        for (var1, var2), label in self.interaction_labels.items():
            if skip_stats:
                self.interactions[label][name] = ""
                continue
            vals1 = self.grouped.get_group(var1)[field]
            vals2 = self.grouped.get_group(var2)[field]
            if (
                not vals1.count()
                or not vals2.count()
                or (np.all(vals1 == 0) and np.all(vals2 == 0))
            ):
                self.interactions[label][name] = "N/A"
                continue
            test = scs.ttest_ind(vals1, vals2, nan_policy="omit")
            df = vals1.count() + vals2.count() - 2
            self.interactions[label][name] = ", ".join(
                [
                    self._format_t(df, test.statistic),
                    self._format_pvalue(test.pvalue),
                ]
            )

    def to_pandas(self):
        col_labels = {label: f"{label} (n={self.n[label]})" for label in self.labels}
        return pd.DataFrame(self.table | self.interactions).rename(columns=col_labels)

    def __repr__(self):
        return repr(self.to_pandas())
