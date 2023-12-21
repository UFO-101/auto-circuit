from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict, List, Optional

from auto_circuit.metrics.answer_diff import measure_answer_diff
from auto_circuit.metrics.answer_diff_percent import measure_answer_diff_percent
from auto_circuit.metrics.answer_value import measure_answer_val
from auto_circuit.metrics.kl_div import measure_kl_div
from auto_circuit.metrics.ROC import measure_roc
from auto_circuit.tasks import Task
from auto_circuit.types import Measurements, MetricKey, PrunedOutputs, PruneScores


@dataclass(frozen=True)
class Metric:
    key: MetricKey
    name: str
    metric_func: Callable[
        [Task, Optional[PruneScores], Optional[PrunedOutputs]], Measurements
    ]
    log_x: bool = False
    log_y: bool = False
    lower_better: bool = False
    y_axes_match: bool = False  # Whether to use the same y-axis for all tasks
    y_min: Optional[float] = None

    def _post_init(self) -> None:
        if self.log_y:
            assert self.y_min is not None


ROC_METRIC = Metric(key="ROC", name="ROC", metric_func=measure_roc)
CLEAN_KL_DIV_METRIC = Metric(
    key="Clean KL Div",
    name="KL Divergence",
    metric_func=partial(measure_kl_div, compare_to_clean=True),
    log_x=True,
    log_y=True,
    lower_better=True,
    y_axes_match=True,
    y_min=1e-1,
)
CORRUPT_KL_DIV_METRIC = Metric(
    key="Corrupt KL Div",
    name="KL Divergence",
    metric_func=partial(measure_kl_div, compare_to_clean=False),
    log_x=True,
    log_y=True,
    lower_better=True,
    y_axes_match=True,
    y_min=1e-1,
)
ANSWER_LOGIT_METRIC = Metric(
    key="Answer Logit",
    name="Answer Logit",
    metric_func=partial(measure_answer_val, prob_func="logits"),
    log_x=True,
    log_y=False,
    lower_better=False,
    y_axes_match=False,
)
ANSWER_PROB_METRIC = Metric(
    key="Answer Prob",
    name="Answer Probability",
    metric_func=partial(measure_answer_val, prob_func="softmax"),
    log_x=True,
    log_y=False,
    lower_better=False,
    y_axes_match=False,
)
ANSWER_LOGPROB_METRIC = Metric(
    key="Answer Logprob",
    name="Answer Log Probability",
    metric_func=partial(measure_answer_val, prob_func="log_softmax"),
    log_x=True,
    log_y=True,
    lower_better=False,
    y_axes_match=False,
    y_min=1e-6,
)
LOGIT_DIFF_METRIC = Metric(
    key="Logit Diff",
    name="Logit Difference",
    metric_func=partial(measure_answer_diff, prob_func="logits"),
    log_x=True,
    log_y=False,
    lower_better=False,
    y_axes_match=False,
)
PROB_DIFF_METRIC = Metric(
    key="Prob Diff",
    name="Probability Difference",
    metric_func=partial(measure_answer_diff, prob_func="softmax"),
    log_x=True,
    log_y=False,
    lower_better=False,
    y_axes_match=False,
)
LOGPROB_DIFF_METRIC = Metric(
    key="Logprob Diff",
    name="Log Probability Difference",
    metric_func=partial(measure_answer_diff, prob_func="log_softmax"),
    log_x=True,
    log_y=True,
    lower_better=False,
    y_axes_match=False,
    y_min=1e-6,
)
LOGIT_DIFF_PERCENT_METRIC = Metric(
    key="Logit Diff Percent",
    name="Logit Difference Percent",
    metric_func=partial(measure_answer_diff_percent, prob_func="logits"),
    log_x=True,
    log_y=False,
    lower_better=False,
    y_axes_match=False,
)
PROB_DIFF_PERCENT_METRIC = Metric(
    key="Prob Diff Percent",
    name="Probability Difference Percent",
    metric_func=partial(measure_answer_diff_percent, prob_func="softmax"),
    log_x=True,
    log_y=False,
    lower_better=False,
    y_axes_match=False,
)
LOBPROB_DIFF_PERCENT_METRIC = Metric(
    key="LogProb Diff Percent",
    name="Log Probability Difference Percent",
    metric_func=partial(measure_answer_diff_percent, prob_func="log_softmax"),
    log_x=True,
    log_y=False,
    lower_better=False,
    y_axes_match=False,
)

METRICS: List[Metric] = [
    ROC_METRIC,
    CLEAN_KL_DIV_METRIC,
    CORRUPT_KL_DIV_METRIC,
    ANSWER_PROB_METRIC,
    ANSWER_LOGIT_METRIC,
    ANSWER_LOGPROB_METRIC,
    LOGIT_DIFF_METRIC,
    PROB_DIFF_METRIC,
    LOGPROB_DIFF_METRIC,
    LOGIT_DIFF_PERCENT_METRIC,
    PROB_DIFF_PERCENT_METRIC,
    LOBPROB_DIFF_PERCENT_METRIC,
]
METRIC_DICT: Dict[MetricKey, Metric] = {metric.key: metric for metric in METRICS}
