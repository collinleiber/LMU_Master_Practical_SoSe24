import csv
import os

import numpy as np
import pandas as pd
import pm4py
import shutil
import uuid
import re
from IPython.utils import io
from datetime import datetime
from pathlib import Path
from sklearn.metrics import silhouette_score

TMP_LOGS_PATH = './tmp_logs'
SAMPLES_PATH = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'example_files'))


def import_csv(file_path):
    event_log = pd.read_csv(file_path, sep=';')
    event_log = pm4py.format_dataframe(event_log, case_id='case_id', activity_key='activity', timestamp_key='timestamp')
    return event_log


def import_xes(file_path):
    event_log = pm4py.read_xes(file_path)
    return event_log


def event_log_to_csv(event_log):
    if os.path.exists(Path(TMP_LOGS_PATH)):
        shutil.rmtree(Path(TMP_LOGS_PATH))
    os.makedirs(TMP_LOGS_PATH, exist_ok=True)
    filename = f'{TMP_LOGS_PATH}/event_log_{uuid.uuid4()}.csv'
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow(["case_id", "activity", "timestamp"])
        for i, activities in enumerate(event_log, start=1):
            for activity in activities:
                timestamp = datetime.now().isoformat()
                writer.writerow([i, activity, timestamp])
    return filename


def read_txt_test_logs(file):
    event_dict = {}
    with open(file, 'r') as file:
        for line in file:
            key, traces = extract_traces_from_text(line)
            event_dict[key] = traces
    return event_dict


def extract_traces_from_text(string_trace):
    extracted_traces = []
    key, traces = re.match(r'(L\d+)\s?=\s?\[(.*)]', string_trace.strip()).groups()
    traces = re.findall(r'<(.*?)>\^(\d+)', traces)
    for trace, frequency in traces:
        for _ in range(int(frequency)):
            extracted_traces.append(tuple(trace.replace(' ', '').split(',')))
    return key, extracted_traces


def deduplicate_list(list_with_duplicates):
    return list(set(list_with_duplicates))


def event_log_to_dataframe(event_log):
    data = []
    for i, activities in enumerate(event_log, start=1):
        for activity in activities:
            timestamp = datetime.now().isoformat()
            data.append([i, activity, timestamp])
    return pd.DataFrame(data, columns=["case_id", "activity", "timestamp"])


def event_log_to_pm4py_dataframe(log):
    """
    Converts the event log to a pandas DataFrame formatted for pm4py.

    Parameters:
        log: List of traces

    Returns:
        A pandas DataFrame with the event log formatted for pm4py.
    """
    data = []
    for i, trace in enumerate(log):
        for event in trace:
            data.append({"case_id": i, "activity": event, "timestamp": i})
    return pm4py.format_dataframe(pd.DataFrame(data), case_id='case_id', activity_key='activity',
                                  timestamp_key='timestamp')


def check_lists_of_sets_equal(list1, list2):
    sorted_list1 = sorted([tuple(sorted(s)) for s in list1])
    sorted_list2 = sorted([tuple(sorted(s)) for s in list2])

    return sorted_list1 == sorted_list2


def custom_metric(log, features, cluster_labels, net, im, fm, weights=None):
    if weights is None:
        weights = {"ss": 0.4, "f": 0.25, "p": 0.35}

    ss = silhouette_score(features, cluster_labels)
    # Calculate and apply penalty for small clusters
    # cluster_sizes = pd.Series(cluster_labels).value_counts(normalize=True)
    # penalty_factor = (cluster_sizes ** 2).sum()
    # ss_adjusted = ss * penalty_factor

    with io.capture_output() as captured:
        fitness = pm4py.conformance.fitness_alignments(log, net, im, fm)["averageFitness"]
        precision = pm4py.conformance.precision_alignments(log, net, im, fm)

    return weights["ss"] * ss + weights["f"] * fitness + weights["p"] * precision


def filter_rare_traces(log, threshold: int = 2, case_column='case:concept:name'):
    """ Filters out traces that occur less than the given threshold in absolut. """
    case_counts = log.groupby(case_column).size()
    frequent_traces = case_counts[case_counts >= threshold].index

    log_filtered = log[log[case_column].isin(frequent_traces)]
    return log_filtered


class Splitter:
    @staticmethod
    def split_by_date(log, date="2018"):
        """
        Splits the log into two parts based on the given date. Filters all cases that are present in both parts.
        """
        date = pd.to_datetime(date).tz_localize('UTC')

        # Split the log
        before = log[log['time:timestamp'] < date]
        after = log[log['time:timestamp'] >= date]

        # Find shared 'case:concept:name' values
        shared_cases = set(before['case:concept:name']).intersection(set(after['case:concept:name']))
        print(f"Shared cases: {len(shared_cases)}")

        # Filter out rows with shared 'case:concept:name' values
        before_filtered = before[~before['case:concept:name'].isin(shared_cases)]
        after_filtered = after[~after['case:concept:name'].isin(shared_cases)]

        return after_filtered, before_filtered

    @staticmethod
    def split_by_cluster(log, cluster_column='cluster'):
        """ Splits the log into sublogs based on the cluster column's unique values. """
        sublogs = {}
        for label in np.unique(log[cluster_column]):
            sublogs[label] = log[log[cluster_column] == label]

        sublogs_sorted = dict(reversed(sorted(
            sublogs.items(), key=lambda item: item[1]['case:concept:name'].nunique()
        )))
        return sublogs_sorted
