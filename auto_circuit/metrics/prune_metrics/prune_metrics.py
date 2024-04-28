from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict, List, Optional

from auto_circuit.data import PromptDataLoader
from auto_circuit.metrics.prune_metrics.answer_diff import measure_answer_diff
from auto_circuit.metrics.prune_metrics.answer_diff_percent import (
    measure_answer_diff_percent,
)
from auto_circuit.metrics.prune_metrics.answer_value import measure_answer_val
from auto_circuit.metrics.prune_metrics.correct_answer_percent import (
    measure_correct_ans_percent,
)
from auto_circuit.metrics.prune_metrics.kl_div import measure_kl_div
from auto_circuit.types import CircuitOutputs, Measurements, PruneMetricKey
from auto_circuit.utils.patchable_model import PatchableModel

"""
PruneMetrics take the outputs of a pruned model and return some measurement of the
model's performance.
"""


@dataclass(frozen=True)
class PruneMetric:
    """
    A metric of the output of a circuit on a task.

    Args:
        key: A unique identifier for the metric.
        name: The name of the metric.
        metric_func: A function that takes a model, a dataloader, and the outputs of
            the ablated model on the dataloader and returns a list of measurements.
        log_x: Whether to log the x-axis when plotting a graph of performance.
        log_y: Whether to log the y-axis when plotting a graph of performance.
        lower_better: Whether lower values are better on the y-axis.
        y_axes_match: Whether to use the same y-axis when plotting multiple tasks.
        y_min: The minimum value for the y-axis when plotting.
    """

    key: PruneMetricKey
    name: str
    metric_func: Callable[
        [PatchableModel, PromptDataLoader, CircuitOutputs], Measurements
    ]
    log_x: bool = False
    log_y: bool = False
    lower_better: bool = False
    y_axes_match: bool = False  # Whether to use the same y-axis for all tasks
    y_min: Optional[float] = None

    def _post_init(self) -> None:
        if self.log_y:
            assert self.y_min is not None


CLEAN_KL_DIV_METRIC = PruneMetric(
    key="Clean KL Div",
    name="KL Divergence",
    metric_func=partial(measure_kl_div, compare_to_clean=True),
    log_x=True,
    log_y=True,
    lower_better=True,
    y_axes_match=False,
    y_min=1e-2,
)
CORRUPT_KL_DIV_METRIC = PruneMetric(
    key="Corrupt KL Div",
    name="KL Divergence",
    metric_func=partial(measure_kl_div, compare_to_clean=False),
    log_x=True,
    log_y=True,
    lower_better=False,
    y_axes_match=False,
    y_min=1e-6,
)
ANSWER_LOGIT_METRIC = PruneMetric(
    key="Answer Logit",
    name="Answer Logit",
    metric_func=partial(measure_answer_val, prob_func="logits", wrong_answer=False),
    log_x=True,
    log_y=False,
    lower_better=False,
    y_axes_match=False,
)
WRONG_ANSWER_LOGIT_METRIC = PruneMetric(
    key="Wrong Answer Logit",
    name="Wrong Answer Logit",
    metric_func=partial(measure_answer_val, prob_func="logits", wrong_answer=True),
    log_x=True,
    log_y=False,
    lower_better=False,
    y_axes_match=False,
)
ANSWER_PROB_METRIC = PruneMetric(
    key="Answer Prob",
    name="Answer Probability",
    metric_func=partial(measure_answer_val, prob_func="softmax", wrong_answer=False),
    log_x=True,
    log_y=False,
    lower_better=False,
    y_axes_match=False,
)
ANSWER_LOGPROB_METRIC = PruneMetric(
    key="Answer Logprob",
    name="Answer Log Probability",
    metric_func=partial(measure_answer_val, prob_func="log_softmax"),
    log_x=True,
    log_y=True,
    lower_better=False,
    y_axes_match=False,
    y_min=1e-6,
)
LOGIT_DIFF_METRIC = PruneMetric(
    key="Logit Diff",
    name="Logit Difference",
    metric_func=partial(measure_answer_diff, prob_func="logits"),
    log_x=True,
    log_y=False,
    lower_better=False,
    y_axes_match=False,
)
PROB_DIFF_METRIC = PruneMetric(
    key="Prob Diff",
    name="Probability Difference",
    metric_func=partial(measure_answer_diff, prob_func="softmax"),
    log_x=True,
    log_y=False,
    lower_better=False,
    y_axes_match=False,
)
LOGPROB_DIFF_METRIC = PruneMetric(
    key="Logprob Diff",
    name="Log Probability Difference",
    metric_func=partial(measure_answer_diff, prob_func="log_softmax"),
    log_x=True,
    log_y=True,
    lower_better=False,
    y_axes_match=False,
    y_min=1e-6,
)
LOGIT_DIFF_PERCENT_METRIC = PruneMetric(
    key="Logit Diff Percent",
    name="Logit Difference Percent",
    metric_func=partial(
        measure_answer_diff_percent, prob_func="logits", diff_of_means=True
    ),
    log_x=True,
    log_y=False,
    lower_better=False,
    y_axes_match=False,
)
PROB_DIFF_PERCENT_METRIC = PruneMetric(
    key="Prob Diff Percent",
    name="Probability Difference Percent",
    metric_func=partial(measure_answer_diff_percent, prob_func="softmax"),
    log_x=True,
    log_y=False,
    lower_better=False,
    y_axes_match=False,
)
LOBPROB_DIFF_PERCENT_METRIC = PruneMetric(
    key="LogProb Diff Percent",
    name="Log Probability Difference Percent",
    metric_func=partial(measure_answer_diff_percent, prob_func="log_softmax"),
    log_x=True,
    log_y=False,
    lower_better=False,
    y_axes_match=False,
)
CORRECT_ANSWER_PERCENT_METRIC = PruneMetric(
    key="Correct Answer Percent",
    name="Correct Answer Percent",
    metric_func=measure_correct_ans_percent,
    log_x=True,
    log_y=False,
    lower_better=False,
    y_axes_match=False,
)
CORRECT_ANSWER_GREATER_THAN_INCORRECT_PERCENT_METRIC = PruneMetric(
    key="Correct Answer Greater Than Incorrect Percent",
    name="Correct Answer Greater Than Incorrect Percent",
    metric_func=partial(
        measure_correct_ans_percent, out_of_correct_and_incorrect_answers=True
    ),
    log_x=True,
    log_y=False,
    lower_better=False,
    y_axes_match=False,
)

PRUNE_METRICS: List[PruneMetric] = [
    CLEAN_KL_DIV_METRIC,
    CORRUPT_KL_DIV_METRIC,
    ANSWER_PROB_METRIC,
    ANSWER_LOGIT_METRIC,
    WRONG_ANSWER_LOGIT_METRIC,
    ANSWER_LOGPROB_METRIC,
    LOGIT_DIFF_METRIC,
    PROB_DIFF_METRIC,
    LOGPROB_DIFF_METRIC,
    LOGIT_DIFF_PERCENT_METRIC,
    PROB_DIFF_PERCENT_METRIC,
    LOBPROB_DIFF_PERCENT_METRIC,
    CORRECT_ANSWER_PERCENT_METRIC,
    CORRECT_ANSWER_GREATER_THAN_INCORRECT_PERCENT_METRIC,
]
PRUNE_METRIC_DICT: Dict[PruneMetricKey, PruneMetric] = {
    metric.key: metric for metric in PRUNE_METRICS
}
