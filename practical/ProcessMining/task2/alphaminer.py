from typing import Dict, List, Tuple, Union
import pandas as pd
import numpy as np
import pm4py
import os


class AlphaMiner:
    """
    The AlphaMiner class implements the alpha miner algorithm for process mining.

    Attributes:
        event_log (pd.DataFrame): The event log data.
        activities (Dict[int, str]): The mapping of activity IDs to activity names.
        all_pairs (List[Tuple[int, int]]): All pairs of activities.
        traces (np.ndarray): The extracted traces from the event log.
        t_in (np.ndarray): The start activities in the traces.
        t_out (np.ndarray): The end activities in the traces.
        following_pairs (np.ndarray): The pairs of activities where the first activity occurs after the second.
        parallel_pairs (np.ndarray): The pairs of activities that are potentially parallel.
        sequential_pairs (np.ndarray): The pairs of activities that are sequential.
        not_following_pairs (np.ndarray): The pairs of activities that do not follow each other.
        before_pairs (np.ndarray): The pairs of activities where the first activity occurs before the second.
    """

    def __init__(self, file_path: str):
        """
        Initializes the AlphaMiner class with a event log file.

        Parameters:
            file_path (str): The path to the event log file.
        """
        self.event_log, self.activities, self.all_pairs = self._import_event_log(file_path)
        self.traces = self._extract_traces(self.event_log)

        self.t_in, self.t_out = self._get_start_end_activities(self.traces)

        self.following_pairs = self._get_following_pairs(self.traces)
        self.parallel_pairs = self._get_parallel_pairs(self.following_pairs)
        self.sequential_pairs = self._get_sequential_pairs(self.following_pairs, self.parallel_pairs)
        self.not_following_pairs = self._get_not_following_pairs(self.following_pairs)
        self.before_pairs = self._get_before_pairs(self.not_following_pairs, self.sequential_pairs, self.parallel_pairs)

    def footprint_matrix(self) -> pd.DataFrame:
        """
        Generates the footprint matrix for the process.

        Returns:
            pd.DataFrame: The footprint matrix.
        """
        matrix = pd.DataFrame(index=sorted(self.activities.values()), columns=sorted(self.activities.values()))

        for pair in self.all_pairs:
            a1, a2 = pair
            a1_value, a2_value = self.activities[a1], self.activities[a2]
            if any(np.array_equal(pair, p) for p in self.parallel_pairs):
                matrix.at[a1_value, a2_value] = '||'
            elif any(np.array_equal(pair, p) for p in self.sequential_pairs):
                matrix.at[a1_value, a2_value] = '→'
            elif any(np.array_equal(pair, p) for p in self.not_following_pairs):
                matrix.at[a1_value, a2_value] = '#'
            elif any(np.array_equal(pair, p) for p in self.before_pairs):
                matrix.at[a1_value, a2_value] = '←'
            else:
                matrix.at[a1_value, a2_value] = ''

        return matrix

    def _import_event_log(self, file_path: str) -> Tuple[pd.DataFrame, Dict[int, str], List[Tuple[int, int]]]:
        """
        Imports the event log from a file.

        Parameters:
            file_path (str): The path to the event log file.

        Returns:
            Tuple[pd.DataFrame, Dict[int, str], List[Tuple[int, int]]]: The event log data,
            the mapping of activity IDs to activity names, and all pairs of activities.
        """
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

        event_log = event_log.sort_values(['case:concept:name', 'time:timestamp'])
        event_log = (event_log[["case:concept:name", "concept:name"]]
                     .rename(columns={"case:concept:name": "case_id", "concept:name": "activity"}))
        activities = self._create_alphabet(event_log)
        all_pairs = [(a1, a2) for a1 in activities.keys() for a2 in activities.keys()]
        event_log['activity_id'] = event_log.apply(lambda row: dict(zip(activities.values(),
                                                                        activities.keys())).get(row['activity']), 1)
        return event_log, activities, all_pairs

    def _create_alphabet(self, event_log: pd.DataFrame) -> Dict[int, str]:
        """
        Creates an alphabet of unique activities.

        Parameters:
            event_log (pd.DataFrame): The event log data.

        Returns:
            Dict[int, str]: The mapping of activity IDs to activity names.
        """
        unique_activities = list(event_log['activity'].unique())
        activities = {i: unique_activities[i] for i in range(0, len(unique_activities))}
        return activities

    def _extract_traces(self, event_log: pd.DataFrame) -> np.ndarray:
        """
        Extracts traces from the event log.

        Parameters:
            event_log (pd.DataFrame): The event log data.

        Returns:
            np.ndarray: The extracted traces.
        """
        traces = np.asarray(event_log.groupby('case_id')['activity_id'].apply(np.asarray).values)
        return traces

    def _get_start_end_activities(self, traces: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Gets the start and end activities from the traces.

        Parameters:
            traces (np.ndarray): The extracted traces.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The start and end activities.
        """
        t_in = np.asarray(list(set([trace[0] for trace in traces])))
        t_out = np.asarray(list(set([trace[-1] for trace in traces])))
        return t_in, t_out

    def _get_following_pairs(self, traces: np.ndarray) -> np.ndarray:
        """
        Gets the pairs of activities that follow each other from the traces.

        Parameters:
            traces (np.ndarray): The extracted traces.

        Returns:
            np.ndarray: The pairs of activities that follow each other like a -> b.
        """
        pairs = []
        for trace in traces:
            for i in range(0, len(trace) - 1):
                if (i + 1) <= len(traces):
                    pairs.append((trace[i], trace[i + 1]))
        unique_pairs = np.asarray(list(set(pairs)))
        return unique_pairs

    def _get_parallel_pairs(self, following_pairs: np.ndarray) -> np.ndarray:
        """
        Gets the pairs of activities that are parallel from the following pairs.

        Parameters:
            following_pairs (np.ndarray): The pairs of activities that follow each other like a -> b.

        Returns:
            np.ndarray: The pairs of activities that are potentially parallel.
        """
        parallel_pairs = []
        reversed_pairs = np.asarray([pair[::-1] for pair in following_pairs])
        for pair in following_pairs:
            for reverse_pair in reversed_pairs:
                if np.array_equal(reverse_pair, pair):
                    parallel_pairs.append(pair)
        return np.asarray(parallel_pairs)

    def _get_sequential_pairs(self, following_pairs: np.ndarray, parallel_pairs: np.ndarray) -> np.ndarray:
        """
        Gets the pairs of activities that are sequential from the following pairs and parallel pairs.

        Parameters:
            following_pairs (np.ndarray): The pairs of activities that follow each other like a -> b.
            parallel_pairs (np.ndarray): The pairs of activities that are potentially parallel.

        Returns:
            np.ndarray: The pairs of activities that are sequential.
        """
        sequential_pairs = np.asarray([pair for pair in following_pairs
                                       if not np.any(np.all(parallel_pairs == pair, axis=1))])
        return sequential_pairs

    def _get_not_following_pairs(self, following_pairs: np.ndarray) -> np.ndarray:
        """
        Gets the pairs of activities that do not follow each other from the following pairs.

        Parameters:
            following_pairs (np.ndarray): The pairs of activities that follow each other like a->b.

        Returns:
            np.ndarray: The pairs of activities that do not follow each other.
        """
        reversed_pairs = np.asarray([pair[::-1] for pair in following_pairs])
        not_following_pairs = np.asarray([pair for pair in self.all_pairs
                                          if not np.any(np.all(following_pairs == pair, axis=1))
                                          and not np.any(np.all(reversed_pairs == pair, axis=1))])
        return not_following_pairs

    def _get_before_pairs(self, not_following_pairs: np.ndarray,
                          sequential_pairs: np.ndarray, parallel_pairs: np.ndarray) -> np.ndarray:
        """
        Gets the pairs of activities where the first activity occurs before the second from the not following pairs,
        sequential pairs, and parallel pairs.

        Parameters:
            not_following_pairs (np.ndarray): The pairs of activities that do not follow each other.
            sequential_pairs (np.ndarray): The pairs of activities that are sequential.
            parallel_pairs (np.ndarray): The pairs of activities that are potentially parallel.

        Returns:
            np.ndarray: The pairs of activities where the first activity occurs before the second.
        """
        before_pairs = np.asarray([pair for pair in self.all_pairs
                                   if not np.any(np.all(not_following_pairs == pair, axis=1))
                                   and not np.any(np.all(sequential_pairs == pair, axis=1))
                                   and not np.any(np.all(parallel_pairs == pair, axis=1))])
        return before_pairs
