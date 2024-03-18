#!/usr/bin/env python3

from gxai.feature_importances import \
    compute_contrast_entropy_importance, \
    compute_contrast_wasserstein_importance, \
    compute_contrast_crowding_importance
from metric_tests import (
    run_category_count_independence_test,
    run_range_independence_test,
    run_middle_value_independence_test,
    run_type_independence_test,
    run_edge_case_behavior_test
)

if __name__ == "__main__":
    importance_score = compute_contrast_crowding_importance

    run_category_count_independence_test(importance_score)
    run_range_independence_test(importance_score)
    run_middle_value_independence_test(importance_score)
    run_type_independence_test(importance_score)
    run_edge_case_behavior_test(importance_score)
