from itertools import combinations
from typing import Dict, List, Tuple, Set
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
        maximal_pairs (np.ndarray): The maximized pair set as result of the alpha miner algorithm.
    """

    def __init__(self, file_path: str, case_id='case_id', activity_key='activity', timestamp_key='timestamp'):
        """
        Initializes the AlphaMiner class with a event log file.

        Parameters:
            file_path (str): The path to the event log file.
        """
        self.event_log, self.activities, self.all_pairs = self._import_event_log(file_path, case_id,
                                                                                 activity_key, timestamp_key)
        self.traces = self._extract_traces(self.event_log)

        self.t_in, self.t_out = self._get_start_end_activities(self.traces)

        self.following_pairs = self._get_following_pairs(self.traces)
        self.parallel_pairs = self._get_parallel_pairs(self.following_pairs)
        self.sequential_pairs = self._get_sequential_pairs(self.following_pairs, self.parallel_pairs)
        self.not_following_pairs = self._get_not_following_pairs(self.following_pairs)
        self.before_pairs = self._get_before_pairs(self.not_following_pairs, self.sequential_pairs, self.parallel_pairs)

        self.maximal_pairs = self._get_maximized_pairs()

    def discover_footprints(self) -> Dict:
        """
        Discovers the footprints of the process from the event log.

        The footprints include:
        - 'dfg': A set of tuples representing the directly-follows graph (DFG). Each tuple contains two activities where the first activity directly follows the second in the process.
        - 'sequence': A set of tuples representing the sequential relations between activities. Each tuple contains two activities where the first activity is directly followed by the second activity in the process.
        - 'parallel': A set of tuples representing the parallel relations between activities. Each tuple contains two activities that can occur in parallel in the process.
        - 'activities': A set of all activities in the process.
        - 'start_activities': A set of activities that can start a process instance.
        - 'end_activities': A set of activities that can end a process instance.
        - 'min_trace_length': The length of the shortest trace in the event log.

        Returns:
            Dict: A dictionary containing the footprints of the process.
        """
        footprints = {
            'dfg': set((self._get_activity_name(pair[0]), self._get_activity_name(pair[1])) for pair in self.following_pairs),
            'sequence': set((self._get_activity_name(pair[0]), self._get_activity_name(pair[1])) for pair in self.sequential_pairs),
            'parallel': set((self._get_activity_name(pair[0]), self._get_activity_name(pair[1])) for pair in self.parallel_pairs),
            'activities': set(self.activities.values()),
            'start_activities': set(self._get_activity_name(activity) for activity in self.t_in),
            'end_activities': set(self._get_activity_name(activity) for activity in self.t_out),
            'min_trace_length': min([len(trace) for trace in self.traces])
        }
        return footprints

    def _get_activity_name(self, activity_id: int) -> str:
        """
        Retrieves the name of an activity based on its ID.

        Parameters:
            activity_id (int): The ID of the activity.

        Returns:
            str: The name of the activity.
        """
        return self.activities[activity_id]

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

    def _import_event_log(self, file_path: str, case_id='case_id', activity_key='activity',
                          timestamp_key='timestamp') -> Tuple[pd.DataFrame, Dict[int, str], List[Tuple[int, int]]]:
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
            event_log = pm4py.format_dataframe(event_log, case_id=case_id, activity_key=activity_key,
                                               timestamp_key=timestamp_key)
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
        unique_activities = sorted(list(event_log['activity'].unique()))
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

    def _get_maximized_pairs(self) -> np.ndarray:
        """
        Iterates over all activities to find maximal pairs and prunes redundant pairs. (Alpha-Algorithm Step 5 & 6)
        xor_split and xor_join are used to store results of the right and left side maximization per activity.
        Result is union of maximal pairs and the difference of sequential pairs without
        sequential pairs used for maximal pairs.

        Returns:
            np.ndarray: The maximized pair set as result of the alpha miner algorithm step 6.
        """
        xor_split, xor_join, result = [], [], []
        for activity in self.activities:
            xor_split.extend(self._right_side_maximization(activity))
            xor_join.extend(self._left_side_maximization(activity))

        result.extend(xor_split), result.extend(xor_join)
        result.extend(self._prune_redundant_sequential_pairs(xor_split, xor_join))

        return np.asarray(list(set(result)), dtype=object)

    def _right_side_maximization(self, activity: int) -> List:
        """
        Maximizes the pairs for the given activity on the right side. (Alpha-Algorithm Step 5)
        For all sequential pairs where the given activity is the first item, all second items are candidates.
        From the power set of these candidates (without reversed items duplicates),
        all pairs that are in not_following_pairs get merged with the activity and added to right side maximized pairs.

        Parameters:
            activity (int): The activity to maximize the pairs for, where it appears as first item.

        Returns:
            np.ndarray: The right side maximized pair set.
        """
        if candidates := sorted([pair[1] for pair in self.sequential_pairs if pair[0] == activity]):
            # When not null, create tuples of each pair that is in not_following_pairs and the given activity
            return [(activity, powered_pair) for powered_pair in list(combinations(candidates, 2))
                    if np.any([np.array_equal(powered_pair, pair) for pair in self.not_following_pairs])]
        return []

    def _left_side_maximization(self, activity: int) -> List:
        """
        Maximizes the pairs for the given activity on the left side. (Alpha-Algorithm Step 5)
        For all sequential pairs where the given activity is the second item, all first items are candidates.
        From the power set of these candidates (without reversed items duplicates),
        all pairs that are in not_following_pairs get merged with the activity and added to left side maximized pairs.

        Parameters:
            activity (int): The activity to maximize the pairs for, where it appears as second item.

        Returns:
            np.ndarray: The left side maximized pair set.
        """
        if candidates := sorted([pair[0] for pair in self.sequential_pairs if pair[1] == activity]):
            # When not null, create tuples of each pair that is in not_following_pairs and the given activity
            return [(powered_pair, activity) for powered_pair in list(combinations(candidates, 2))
                    if np.any([np.array_equal(powered_pair, pair) for pair in self.not_following_pairs])]
        return []

    def _prune_redundant_sequential_pairs(self, split_result: List[Tuple], join_result: List[Tuple]) -> np.ndarray:
        """
        Prunes redundant pairs from the sequential pairs. (Alpha-Algorithm Step 6)
        When a maximal pair (y, z) appears in split_result or join_results as first or second item,
        all sequential pairs (x, y) and (x, z) or (y, x) and (z, x) get removed.
        Returns all remaining pairs from sequential_pairs, which were accordingly not used to find maximal pairs.

        Parameters:
            split_result (List[Tuple]): The result set of right side maximization.
            join_result (List[Tuple]): The result set of left side maximization.

        Returns:
            np.ndarray: The set of sequential pairs that are not redundant.
        """
        minimal_pairs = np.copy(self.sequential_pairs)

        if minimal_pairs.any():
            # Remove entries (x, y) and (x, z) from stack, when (x, (y, z)) is in split_result
            for x, (y, z) in split_result:
                minimal_pairs = minimal_pairs[
                    ~((minimal_pairs[:, 0] == x) & ((minimal_pairs[:, 1] == y) | (minimal_pairs[:, 1] == z)))
                ]

        if minimal_pairs.any():
            # Remove entries (y, x) and (z, x) from stack, when ((y, z), x) is in join_result
            for (y, z), x in join_result:
                minimal_pairs = minimal_pairs[
                    ~((minimal_pairs[:, 1] == x) & ((minimal_pairs[:, 0] == y) | (minimal_pairs[:, 0] == z)))
                ]

        return minimal_pairs

    def get_maximal_pairs(self) -> List[Tuple[Set[str], Set[str]]]:
        """
        Returns the maximized pair set as result of the alpha miner algorithm.

        Returns:
            np.ndarray: The maximized pair set.
        """
        return self._activity_encoder(self.maximal_pairs, getter=True)

    def print_pairs(self, encoded: bool = True) -> None:
        """
        Debugging method.
        Prints all pair types of current alpha miner instance.

        Parameters:
            encoded (bool): Whether to print the pairs with activity IDs or names.
        """
        self._activity_encoder(self.following_pairs, "Following pairs", encoded=encoded)
        self._activity_encoder(self.parallel_pairs, "Parallel pairs", encoded=encoded)
        self._activity_encoder(self.sequential_pairs, "Sequential pairs", encoded=encoded)
        self._activity_encoder(self.not_following_pairs, "Not following pairs", encoded=encoded)
        self._activity_encoder(self.before_pairs, "Before pairs", encoded=encoded)
        self._activity_encoder(self.maximal_pairs, "Maximal pairs", encoded=encoded)

    def print_single_pair_type(self, pair_type: str = ">", encoded: bool = True) -> None:
        """
        Debugging method.

        Parameters:
            pair_type (str): The pair type to print. Options are: ">", "||", "->", "#", "<-", "max".
            encoded (bool): Whether to print the pairs with activity IDs or names.
        """
        if pair_type == ">":
            self._activity_encoder(self.following_pairs, "Following pairs", encoded=encoded)
        elif pair_type == "||":
            self._activity_encoder(self.parallel_pairs, "Parallel pairs", encoded=encoded)
        elif pair_type == "->":
            self._activity_encoder(self.sequential_pairs, "Sequential pairs", encoded=encoded)
        elif pair_type == "#":
            self._activity_encoder(self.not_following_pairs, "Not following pairs", encoded=encoded)
        elif pair_type == "<-":
            self._activity_encoder(self.before_pairs, "Before pairs", encoded=encoded)
        elif pair_type == "max":
            self._activity_encoder(self.maximal_pairs, "Maximal pairs", encoded=encoded)

    def _activity_encoder(self, pairs: np.ndarray, description: str = "",
                          encoded: bool = True, getter: bool = False) -> None or List[Tuple[Set, Set]]:
        """
        Helper method to print pairs with activity IDs or names.

        Parameters:
            pairs (np.ndarray): The pairs to print.
            description (str): The description, naming the pair type.
            encoded (bool): Whether to print the pairs with activity IDs or names.
        """
        output = []
        alphabet = self.activities
        int_only = True

        if encoded:
            for pair in pairs:
                first = pair[0]
                second = pair[1]

                if isinstance(first, tuple):
                    first = {alphabet.get(first[0]), alphabet.get(first[1])}
                    int_only = False
                else:
                    first = {alphabet.get(first)}
                if isinstance(second, tuple):
                    second = {alphabet.get(second[0]), alphabet.get(second[1])}
                    int_only = False
                else:
                    second = {alphabet.get(second)}
                output.append((first, second))

        if not getter:
            print(description + ":")
            print(sorted(output)) if int_only else print(output)
            print()
        else:
            return output
