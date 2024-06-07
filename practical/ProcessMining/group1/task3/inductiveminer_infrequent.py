from collections import defaultdict
from typing import Optional, List, Tuple, Dict, Set
from practical.ProcessMining.group1.task3.inductiveminer import InductiveMiner, CutType
from practical.ProcessMining.group1.shared.utils import deduplicate_list


class InductiveMinerInfrequent(InductiveMiner):
    def __init__(self, event_log: Optional[List[Tuple[str]]] = None, threshold: float = 0.0):
        super().__init__(event_log=event_log)
        self.threshold = threshold

    def run(self) -> None:
        # Initialize the list of sublogs with the original event log
        sublogs = [self.event_log]

        # Iterate over the sublogs until the list is empty
        while len(sublogs) > 0:
            log = sublogs[0]

            # Update the directly-follows graph (dfg), start_activities, and end_activities for the current sublog
            dfg, start_activities, end_activities = self._get_dfg(log)

            # Check for base cases and build the corresponding part of the process tree
            base_cut, operator = self._handle_base_cases(log)
            if base_cut:  # If not a base case, apply different types of cuts
                self._build_process_tree(base_cut, operator)
            else:  # try to split log based on operator and build corresponding part of the process tree
                groups, operator = self._apply_cut(log, dfg, start_activities, end_activities)
                # Add the new sublogs to the list if not fall through case
                if operator != CutType.NONE:
                    new_sublogs = self._split_log(log, groups)
                    sublogs.extend(new_sublogs)
                    # Build the corresponding part of the process tree
                    self._build_process_tree(groups, operator)
                else:  # If fall through case, apply infrequent logic TODO refactor for super call
                    base_cut, operator = self._handle_base_cases_filtered(log)
                    if base_cut:  # If not a base case, apply different types of cuts
                        self._build_process_tree(base_cut, operator)
                    else:  # try to split log based on operator and build corresponding part of the process tree
                        groups, operator = self._apply_cut_filtered(log, dfg, start_activities, end_activities)
                        if operator != CutType.NONE:  # If not fall through case
                            new_sublogs = self._split_log_filtered(log, groups, operator)  # Apply IMi filters
                            sublogs.extend(new_sublogs)
                            self._build_process_tree(groups, operator)
                        else:  # If fall through case, apply flower model
                            self._build_process_tree(groups, CutType.LOOP)

            # Remove the old sublog from the list
            sublogs.remove(log)

    def _apply_cut_filtered(self, log: List[Tuple[str]], dfg: Dict[Tuple[str, str], int],
                            start_activities: Dict[str, int], end_activities: Dict[str, int]) \
            -> Tuple[List[Set[str]], CutType]:

        dfg_filtered = self.get_frequent_directly_follows_graph(dfg)
        efg_filtered = self.get_frequent_eventually_follows_graph(dfg)

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

    def get_frequent_directly_follows_graph(self, dfg):
        max_freq = defaultdict(int)
        for edge, frequency in dfg.items():
            max_freq[edge[0]] = max(max_freq[edge[0]], frequency)

        # Filter infrequent edges of compared to max edge of each node
        frequent_dfg = {edge: frequency for edge, frequency in dfg.items()
                        if frequency >= self.threshold * max_freq[edge[0]]}

        # Sort by frequency
        frequent_dfg = dict(sorted(frequent_dfg.items(), key=lambda item: item[1], reverse=True))
        return frequent_dfg

    def _calculate_eventually_follows_graph(self, dfg):
        efg = dfg.copy()
        # Repeat until no more edges can be added
        while True:
            # Track whether a new edge is added in this iteration
            new_edge_added = False

            # Iterate over each pair of nodes in the efg
            for (i, j), freq_ij in list(efg.items()):
                for (k, l), freq_kl in list(efg.items()):
                    # If there is a path from i to l through j and k, add an edge (i, l) to the efg
                    if j == k and (i, l) not in efg:
                        efg[(i, l)] = min(freq_ij,
                                          freq_kl)  # The frequency is the minimum of the frequencies of the two edges
                        new_edge_added = True

            # If no new edge is added in this iteration, break the loop
            if not new_edge_added:
                break

        return efg

    def get_frequent_eventually_follows_graph(self, dfg) -> Dict[Tuple[str, str], int]:
        efg = self._calculate_eventually_follows_graph(dfg)

        return self.get_frequent_directly_follows_graph(efg)

    def _split_log_filtered(self, log: List[Tuple[str]], groups: List[Set[str]],
                            operator: CutType) -> List[List[Tuple[str]]]:
        # TODO: Apply IMi filters on log splitting based on operator
        return []

    def xor_split_infrequent(self):
        pass

    def sequence_split_infrequent(self):
        pass

    def loop_split_infrequent(self):
        pass

