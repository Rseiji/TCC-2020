import pandas as pd
import numpy as np
import csv
from pandas.core.frame import DataFrame
from labels import generate_labels
from segments import generate_segment_ids


def get_scores_from_base_dataset(base_dataset: DataFrame) -> list:
    return base_dataset['score'].to_list()


def get_comments_from_base_dataset(base_dataset: DataFrame) -> list:
    return base_dataset['comment'].to_list()


def build_full_dataset(base_dataset: DataFrame, segment_ids: list, labels: list) -> DataFrame:

    df_with_segmentIds = base_dataset.assign(segment_id=segment_ids)
    return df_with_segmentIds.assign(label=labels)


def get_base_dataset(path_file: str) -> DataFrame:
    try:
        df = pd.read_csv(path_file, header=0,
                         lineterminator='\n', skipinitialspace=True)

        return df.replace(np.NaN, '')

    except Exception as error:
        print(f'Error reading the file. Error: {error}')


def save_full_dataset_to_file(full_dataset: DataFrame, path_file: str):
    full_dataset.to_csv(
        path_file, index=False, quoting=csv.QUOTE_NONNUMERIC)


def generate_full_dataset():
    path_file = './base_dataset.csv'

    base_dataset = get_base_dataset(path_file)

    comments = get_comments_from_base_dataset(base_dataset)

    segment_ids = generate_segment_ids(comments)

    scores = get_scores_from_base_dataset(base_dataset)

    labels = generate_labels(scores)

    full_dataset = build_full_dataset(base_dataset, segment_ids, labels)

    save_full_dataset_to_file(
        full_dataset, 'full_dataset_segment_last_comment_label_100.csv')
