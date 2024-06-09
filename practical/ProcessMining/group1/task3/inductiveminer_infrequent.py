from collections import defaultdict
from typing import Optional, List, Tuple, Dict, Set

from practical.ProcessMining.group1.task3.inductiveminer import InductiveMiner, CutType
from practical.ProcessMining.group1.shared.utils import deduplicate_list


class InductiveMinerInfrequent(InductiveMiner):
    """
    Inductive Miner infrequent implementation based on the paper:
    "Process Mining: Discovery, Conformance and Enhancement of Business Processes" by Wil M.P. van der Aalst

    Attributes (which are not inherited):
        threshold: coefficient used to define infrequency. (1 - threshold) * 100 => outliers
    """
    def __init__(self, event_log: Optional[List[Tuple[str]]] = None, threshold: float = 0.0):
        super().__init__(event_log=event_log)
        self.threshold = threshold

    def run(self) -> None:
        """
        Main method that start recursive process tree discovery / building.
        """

        # Initialize the list of sublogs with the original event log
        sublogs = [self.event_log]

        # Iterate over the sublogs until the list is empty
        while len(sublogs) > 0:
            log = sublogs[0]
            # Run basic inductive miner recursion step
            result, groups, new_sublogs = super().recursion_step(log)

            # When no result, run IMi recursion step
            if not result:
                new_sublogs = self.recursion_step(log)

            # Update sublogs
            sublogs.extend(new_sublogs) if new_sublogs else sublogs
            sublogs.remove(log)

    def recursion_step(self, log: List[Tuple[str]]) -> List[List[Tuple[str]]]:
        """
        Single recursion step of Inductive Miner infrequent. Only gets called, when super method found no cut.

        Parameters:
            log: sublog as subset of the original event log
        """

        # Update the directly-follows graph (dfg), start_activities, and end_activities for the current sublog
        dfg, start_activities, end_activities = self._get_dfg(log)
        new_sublogs = []

        base_cut, operator = self._handle_base_cases_filtered(log)
        if base_cut:  # If not a base case, apply different types of cuts
            self._build_process_tree(base_cut, operator)
        else:  # try to split log based on operator and build corresponding part of the process tree
            groups, operator = self._apply_cut_filtered(log, dfg, start_activities, end_activities)
            if operator != CutType.NONE:  # If not fall through case
                new_sublogs = self._split_log_filtered(log, groups, operator)  # Apply IMi filters
                self._build_process_tree(groups, operator)
            else:  # If fall through case, apply flower model
                self._build_process_tree(groups, CutType.LOOP)

        return new_sublogs

    def _apply_cut_filtered(self, log: List[Tuple[str]], dfg: Dict[Tuple[str, str], int],
                            start_activities: Dict[str, int],
                            end_activities: Dict[str, int]) -> Tuple[List[Set[str]], CutType]:

        dfg_filtered = self.get_frequent_directly_follows_graph(dfg)
        efg_filtered = self.get_frequent_eventually_follows_graph(log)

        # TODO refactor apply_cut to make super call possible instead of duplicate code
        # super()._apply_cut(log=log, dfg=dfg, start_activities=dfg_start, end_activities=dfg_end)

        # Try to apply different types of cuts to the current sublog
        sequence_cut = self._sequence_cut(efg_filtered, start_activities, end_activities)
        xor_cut = self._xor_cut(dfg_filtered, start_activities, end_activities)
        parallel_cut = self._parallel_cut(dfg_filtered, start_activities, end_activities)
        loop_cut = self._loop_cut(dfg_filtered, start_activities, end_activities)

        # If a nontrivial cut (>1) is found, return the partition and the corresponding operator
        if self._is_nontrivial(sequence_cut):
            return sequence_cut, CutType.SEQUENCE
        elif self._is_nontrivial(xor_cut):
            return xor_cut, CutType.XOR
        elif self._is_nontrivial(parallel_cut):
            return parallel_cut, CutType.PARALLEL
        elif self._is_nontrivial(loop_cut):
            return loop_cut, CutType.LOOP
        else:  # If no nontrivial cut is found, apply the fall-through case (flower model)
            flower_groups = self._handle_fall_through(log)
            return flower_groups, CutType.NONE

    def _handle_base_cases_filtered(self, log: List[Tuple[str]]) -> Tuple[List[Set[str]], CutType]:
        # filter single activities
        log = self._single_activity_filtering(log)
        # filter empty traces
        log = self._empty_trace_filtering(log)
        # Apply the base case logic on the filtered log
        return super()._handle_base_cases(log)

    def _single_activity_filtering(self, log: List[Tuple[str]]) -> List[Tuple[str]]:
        # Get all traces in the log that contain only a single activity and are not empty
        single_activity_traces = [trace for trace in log if len(trace) == 1 and trace[0] != '']

        # If all single activity traces are the same (otherwise no base case)
        if len(deduplicate_list(single_activity_traces)) == 1:
            # Calculate the relative frequency of the single activity traces in the log
            rel_freq = sum([1 for trace in log if trace == single_activity_traces[0]]) / len(log)
            # If the relative frequency of the single activity traces is above the threshold
            if 1 - rel_freq <= self.threshold:
                # Filter out all traces that are not the single activity trace
                filtered_traces = [trace for trace in log if trace == single_activity_traces[0]]
                return filtered_traces

        return log  # If no filtering was applied, return the original log

    def _empty_trace_filtering(self, log: List[Tuple[str]]) -> List[Tuple[str]]:
        # Get all traces in the log that are empty
        empty_traces = [trace for trace in log if trace == ('',)]

        # Calculate the relative frequency of the empty traces in the log
        rel_freq = len(empty_traces) / len(log)
        # If the relative frequency of the empty traces is below the threshold
        if rel_freq <= self.threshold:
            # Filter out all traces that are empty
            filtered_traces = [trace for trace in log if trace != ('',)]
            return filtered_traces

        return log  # If no filtering was applied, return the original log

    def get_frequent_directly_follows_graph(self, dfg: Dict[Tuple[str, str], int]):
        """
        Takes as input current directly follows graph and for each node,
        all edges get removed as infrequent relatively compared to the most frequent outgoing edge
        multiplied by a given threshold.

        Parameters:
            dfg: current directly follows graph
        """
        max_freq = defaultdict(int)
        for edge, frequency in dfg.items():
            max_freq[edge[0]] = max(max_freq[edge[0]], frequency)

        # Filter infrequent edges of compared to max edge of each node
        frequent_dfg = {edge: frequency for edge, frequency in dfg.items()
                        if frequency >= self.threshold * max_freq[edge[0]]}

        # Sort by frequency
        frequent_dfg = dict(sorted(frequent_dfg.items(), key=lambda item: item[1], reverse=True))
        return frequent_dfg

    def _calculate_eventually_follows_graph(self, log: List[Tuple[str]]) -> Dict[Tuple[str, str], int]:
        """
        From a given sublog, an eventually follows graph is created covering not only directly follows pairs
        but all transitive related follows pairs. Used for the filtered sequence cut.

        Parameters:
            log: sublog as subset of the original event log
        """
        efg = defaultdict(int)

        for trace in log:
            for first in range(len(trace) - 1):
                for second in range(first + 1, len(trace)):
                    pair = (trace[first], trace[second])
                    efg[pair] += 1

        return efg

    def get_frequent_eventually_follows_graph(self, log) -> Dict[Tuple[str, str], int]:
        """
        Calculates eventually follows graph and filters according to dfg infrequent edges.

        Parameters:
            log: sublog as subset of the original event log
        """
        efg = self._calculate_eventually_follows_graph(log)

        # Since efg and dfg have the same data structure, dfg named method reused for filtering
        return self.get_frequent_directly_follows_graph(efg)

    def _split_log_filtered(self, log: List[Tuple[str]], groups: List[Set[str]],
                            operator: CutType) -> List[List[Tuple[str, ...]]]:
        """
        Selects the correct splitting method based on the given operator.

        Parameters:
            log: (sub)log as subset of event log
            groups: list of languages as result of applied cut
            operator: enum containing type of cut
        """
        if operator == CutType.SEQUENCE:
            return self.sequence_split_filtered(log, groups)
        elif operator == CutType.XOR:
            return self.xor_split_filtered(log, groups)
        elif operator == CutType.LOOP:
            return self.loop_split_filtered(log, groups)
        elif operator == CutType.PARALLEL:
            return super()._parallel_split(log, groups)
        return []

    def xor_split_frequent(self, log: List[Tuple[str]], groups: List[Set[str]]):
        sublogs = [[] for _ in range(len(groups))]

        for trace in log:
            min_freq = len(trace) * self.threshold

            for i, group in enumerate(groups):

                sub_trace = tuple(activity for activity in trace if activity in group)
                if sub_trace:
                    sublogs[i].append(sub_trace)
                    break

        sublogs = [sublog for sublog in sublogs if sublog]
        return sublogs

    def sequence_split_frequent(self, log: List[Tuple[str]], groups: List[Set[str]]):
        pass

    def loop_split_frequent(self, log: List[Tuple[str]], groups: List[Set[str]]):
        pass
