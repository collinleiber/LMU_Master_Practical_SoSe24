import csv
import os
import pandas as pd
import pm4py
import shutil
import uuid
import re
from datetime import datetime
from pathlib import Path

TMP_LOGS_PATH = './tmp_logs'


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
            key, traces = re.match(r'(L\d+) = \[(.*)\]', line.strip()).groups()
            traces = re.findall(r'<(.*?)>\^(\d+)', traces)
            event_dict[key] = []
            for trace, frequency in traces:
                for _ in range(int(frequency)):
                    event_dict[key].append(tuple(trace.replace(' ', '').split(',')))
    return event_dict


def deduplicate_list(list_with_duplicates):
    return list(set(list_with_duplicates))


def event_log_to_dataframe(event_log):
    data = []
    for i, activities in enumerate(event_log, start=1):
        for activity in activities:
            timestamp = datetime.now().isoformat()
            data.append([i, activity, timestamp])
    return pd.DataFrame(data, columns=["case_id", "activity", "timestamp"])


def check_lists_of_sets_equal(list1, list2):
    sorted_list1 = sorted([tuple(sorted(s)) for s in list1])
    sorted_list2 = sorted([tuple(sorted(s)) for s in list2])

    return sorted_list1 == sorted_list2
