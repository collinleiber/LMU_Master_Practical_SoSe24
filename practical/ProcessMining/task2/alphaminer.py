import sys

import pandas as pd
import pm4py
import os


class AlphaMiner:

    def __init__(self, file_path):
        self.event_log = self._import_event_log(file_path)
        self.traces = self._extract_traces(self.event_log)

    def _import_event_log(self, file_path) -> pd.DataFrame:
        if not os.path.exists(file_path):
            raise Exception("File does not exist")

        extension = os.path.splitext(file_path)[1]
        if extension == '.csv':
            event_log = pd.read_csv(file_path, sep=';')
            event_log = pm4py.format_dataframe(event_log, case_id='case_id', activity_key='activity',
                                               timestamp_key='timestamp')
        elif extension == '.xes':
            event_log = pm4py.read_xes(file_path)
        else:
            raise Exception("File extension must be .csv or xes")

        event_log = event_log[["case:concept:name", "concept:name"]].rename(columns={"case:concept:name": "case_id",
                                                                                     "concept:name": "activity"})
        return event_log

    def _extract_traces(self, event_log):
        return event_log.groupby('case_id')['activity'].apply(list).values.tolist()


