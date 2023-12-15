from functools import partial

from auto_circuit.metrics.answer_diff import measure_answer_diff
from auto_circuit.metrics.answer_diff_percent import measure_answer_diff_percent
from auto_circuit.metrics.answer_value import measure_answer_val
from auto_circuit.metrics.kl_div import measure_kl_div
from auto_circuit.metrics.ROC import measure_roc
from auto_circuit.types import Metric

ROC_METRIC = Metric("ROC", measure_roc)
CLEAN_KL_DIV_METRIC = Metric(
    "KL Divergence",
    partial(measure_kl_div, compare_to_clean=True),
    log_x=True,
    log_y=True,
    lower_better=True,
    y_axes_match=True,
    y_min=1e-1,
)
CORRUPT_KL_DIV_METRIC = Metric(
    "KL Divergence",
    partial(measure_kl_div, compare_to_clean=False),
    log_x=True,
    log_y=True,
    lower_better=True,
    y_axes_match=True,
    y_min=1e-1,
)
ANSWER_LOGIT_METRIC = Metric(
    "Answer Logit",
    partial(measure_answer_val, prob_func="logits"),
    log_x=True,
    log_y=False,
    lower_better=False,
    y_axes_match=False,
)
ANSWER_PROB_METRIC = Metric(
    "Answer Probability",
    partial(measure_answer_val, prob_func="softmax"),
    log_x=True,
    log_y=False,
    lower_better=False,
    y_axes_match=False,
)
ANSWER_LOGPROB_METRIC = Metric(
    "Answer Log Probability",
    partial(measure_answer_val, prob_func="log_softmax"),
    log_x=True,
    log_y=True,
    lower_better=False,
    y_axes_match=False,
    y_min=1e-6,
)
LOGIT_DIFF_METRIC = Metric(
    "Logit Difference",
    partial(measure_answer_diff, prob_func="logits"),
    log_x=True,
    log_y=False,
    lower_better=False,
    y_axes_match=False,
)
PROB_DIFF_METRIC = Metric(
    "Probability Difference",
    partial(measure_answer_diff, prob_func="softmax"),
    log_x=True,
    log_y=False,
    lower_better=False,
    y_axes_match=False,
)
LOGPROB_DIFF_METRIC = Metric(
    "Log Probability Difference",
    partial(measure_answer_diff, prob_func="log_softmax"),
    log_x=True,
    log_y=True,
    lower_better=False,
    y_axes_match=False,
    y_min=1e-6,
)
LOGIT_DIFF_PERCENT_METRIC = Metric(
    "Logit Difference Percent",
    partial(measure_answer_diff_percent, prob_func="logits"),
    log_x=True,
    log_y=False,
    lower_better=False,
    y_axes_match=False,
)
PROB_DIFF_PERCENT_METRIC = Metric(
    "Probability Difference Percent",
    partial(measure_answer_diff_percent, prob_func="softmax"),
    log_x=True,
    log_y=False,
    lower_better=False,
    y_axes_match=False,
)
LOBPROB_DIFF_PERCENT_METRIC = Metric(
    "Log Probability Difference Percent",
    partial(measure_answer_diff_percent, prob_func="log_softmax"),
    log_x=True,
    log_y=False,
    lower_better=False,
    y_axes_match=False,
)
