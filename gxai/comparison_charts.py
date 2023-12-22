#!/usr/bin/env python3

from matplotlib import pyplot as plt
import os
import pandas as pd
from typing import List

from .feature_types import FeatureType, CategoricalFeature, ContinuousFeature


def generate_comparison_charts(
    data1: pd.DataFrame,
    data2: pd.DataFrame,
    feature_types: List[FeatureType],
    target_dir: str = "tmp",
    data1_title: str = "Original Population",
    data2_title: str = "Improved Population",
) -> None:
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for column_name, feature_type in zip(data1.columns, feature_types):
        figure, axes = plt.subplots(1, 2, sharey=True)
        figure.suptitle(column_name)

        if feature_type == CategoricalFeature:
            # do not use feature_type.unique_values here, since the values might
            # have been reverse mapped
            unique_feature_values = (
                pd.concat([data1[column_name], data2[column_name]]).unique().tolist()
            )

            feature_counts1 = [0] * len(unique_feature_values)
            feature_counts2 = [0] * len(unique_feature_values)

            for x1_i in data1[column_name].tolist():
                heights_i = unique_feature_values.index(x1_i)
                feature_counts1[heights_i] += 1

            for x2_i in data2[column_name].tolist():
                heights_i = unique_feature_values.index(x2_i)
                feature_counts2[heights_i] += 1

            heights1 = [count / sum(feature_counts1) for count in feature_counts1]

            heights2 = [count / sum(feature_counts2) for count in feature_counts2]

            unique_feature_values = [str(value) for value in unique_feature_values]
            axes[0].bar(x=unique_feature_values, height=heights1)
            axes[0].set_title(data1_title)

            axes[1].bar(x=unique_feature_values, height=heights2)
            axes[1].set_title(data2_title)

        elif feature_type == ContinuousFeature:
            min_feature_value = pd.concat(
                [data1[column_name], data2[column_name]]
            ).min()
            max_feature_value = pd.concat(
                [data1[column_name], data2[column_name]]
            ).max()

            bin_count = 10

            axes[0].hist(
                x=data1[column_name].tolist(),
                bins=bin_count,
                range=[min_feature_value, max_feature_value],
                density=True,
                rwidth=0.95,
            )
            axes[0].set_title(data1_title)

            axes[1].hist(
                x=data2[column_name].tolist(),
                bins=bin_count,
                range=[min_feature_value, max_feature_value],
                density=True,
                rwidth=0.95,
            )
            axes[1].set_title(data2_title)
        else:
            raise NotImplementedError()

        target_path = f"{target_dir}/{column_name}.png"
        figure.savefig(target_path)
        plt.close()
