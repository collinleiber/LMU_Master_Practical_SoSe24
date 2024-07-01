import csv
import os
import pandas as pd
import pm4py
import shutil
import uuid
import re
from datetime import datetime
from pathlib import Path
import xml.etree.ElementTree as ElementTree

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


def cleanse_real_eventlog(input_file_path=None, output_file_path=None):
    if not input_file_path:
        input_file_path = SAMPLES_PATH / 'DomesticDeclarations.xes'
        output_file_path = SAMPLES_PATH / 'DomesticDeclarations_cleansed.csv'

    tree = ElementTree.parse(input_file_path)
    root = tree.getroot()

    with open(output_file_path, mode='w', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        # Write the CSV file header
        writer.writerow(['case_id', 'timestamp', 'activity'])

        # Iterate over each trace
        for trace in root.findall('trace'):
            case_id = trace.find('string[@key="id"]').get('value')

            # Iterate over each event
            for event in trace.findall('event'):
                timestamp = event.find('date[@key="time:timestamp"]').get('value')
                activity = event.find('string[@key="concept:name"]').get('value')

                # Write to CSV file
                writer.writerow([case_id, timestamp, activity])

    print("Done")
