import pandas as pd
import numpy as np
import pm4py
import os


class AlphaMiner:

    def __init__(self, file_path):
        self.event_log, self.activities = self._import_event_log(file_path)
        self.traces = self._extract_traces(self.event_log)

        self.t_in, self.t_out = self._get_start_end_activities(self.traces)

        self.following_pairs = self._get_following_pairs(self.traces)
        self.parallel_pairs = self._get_parallel_pairs(self.following_pairs)
        self.sequential_pairs = self._get_sequential_pairs(self.following_pairs, self.parallel_pairs)
        self.not_following_pairs = self._get_not_following_pairs(self.following_pairs)
        self.before_pairs = self._get_before_pairs(self.not_following_pairs)

    def _import_event_log(self, file_path) -> tuple[pd.DataFrame, dict]:
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

        event_log = (event_log[["case:concept:name", "concept:name"]]
                     .rename(columns={"case:concept:name": "case_id", "concept:name": "activity"}))
        activities = self._create_alphabet(event_log)
        event_log['activity_id'] = event_log.apply(lambda row: dict(zip(activities.values(),
                                                                        activities.keys())).get(row['activity']), 1)
        return event_log, activities

    def _create_alphabet(self, event_log):
        unique_activities = list(event_log['activity'].unique())
        activities = {i: unique_activities[i] for i in range(0, len(unique_activities))}
        return activities

    def _extract_traces(self, event_log):
        return np.asarray(event_log.groupby('case_id')['activity_id'].apply(np.asarray).values)

    def _get_start_end_activities(self, traces):
        t_in = np.asarray(list(set([trace[0] for trace in traces])))
        t_out = np.asarray(list(set([trace[-1] for trace in traces])))
        return t_in, t_out

    def _get_following_pairs(self, traces):
        pairs = []
        for trace in traces:
            for i in range(0, len(trace)):
                if i < len(traces) - 2:
                    pairs.append((trace[i], trace[i + 1]))
        unique_pairs = np.asarray(list(set(pairs)))
        return unique_pairs

    def _get_parallel_pairs(self, following_pairs):
        parallel_pairs = []
        reversed_pairs = np.asarray([pair[::-1] for pair in following_pairs])
        for pair in following_pairs:
            for reverse_pair in reversed_pairs:
                if np.array_equal(reverse_pair, pair):
                    parallel_pairs.append(pair)
        return np.asarray(parallel_pairs)

    def _get_sequential_pairs(self, following_pairs, parallel_pairs):
        sequential_pairs = np.asarray([pair for pair in following_pairs
                                       if not np.any(np.all(parallel_pairs == pair, axis=1))])
        return sequential_pairs

    def _get_not_following_pairs(self, following_pairs):
        all_pairs = []
        for a1 in self.activities.keys():
            for a2 in self.activities.keys():
                all_pairs.append((a1, a2))
        all_pairs = np.asarray(list(set(all_pairs)))
        reversed_pairs = np.asarray([pair[::-1] for pair in following_pairs])
        not_following_pairs = np.asarray([pair for pair in all_pairs
                                          if not np.any(np.all(following_pairs == pair, axis=1))
                                          and not np.any(np.all(reversed_pairs == pair, axis=1))])
        return not_following_pairs

    def _get_before_pairs(self, not_following_pairs):
        all_pairs = []
        for a1 in self.activities.keys():
            for a2 in self.activities.keys():
                all_pairs.append((a1, a2))
        all_pairs = np.asarray(list(set(all_pairs)))
        not_following_pairs = np.asarray([pair for pair in all_pairs
                                          if not np.any(np.all(not_following_pairs == pair, axis=1))])
        return not_following_pairs
