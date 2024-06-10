from collections import defaultdict
from itertools import combinations, chain
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
        """
        Apply different types of cuts to the current sublog based on the filtered directly follows graph and
        eventually follows graph.

        Parameters:
            log: sublog as subset of the original event log
            dfg: current directly follows graph
            start_activities: start activities of the current sublog
            end_activities: end activities of the current sublog

        Returns:
            Tuple of groups and operator
        """
        dfg_filtered = self.get_frequent_directly_follows_graph(dfg)
        efg_filtered = self.get_frequent_eventually_follows_graph(log)

        # TODO refactor apply_cut to make super call possible instead of duplicate code
        # super()._apply_cut(log=log, dfg=dfg, start_activities=dfg_start, end_activities=dfg_end)

        # Try to apply different types of cuts to the current sublog
        xor_cut = self._xor_cut(dfg_filtered, start_activities, end_activities)
        sequence_cut = self._sequence_cut(efg_filtered, start_activities, end_activities)
        parallel_cut = self._parallel_cut(dfg_filtered, start_activities, end_activities)
        loop_cut = self._loop_cut(dfg_filtered, start_activities, end_activities)

        # If a nontrivial cut (>1) is found, return the partition and the corresponding operator
        if self._is_nontrivial(xor_cut):
            return xor_cut, CutType.XOR
        elif self._is_nontrivial(sequence_cut):
            return sequence_cut, CutType.SEQUENCE
        elif self._is_nontrivial(parallel_cut):
            return parallel_cut, CutType.PARALLEL
        elif self._is_nontrivial(loop_cut):
            return loop_cut, CutType.LOOP
        else:  # If no nontrivial cut is found, apply the fall-through case (flower model)
            flower_groups = self._handle_fall_through(log)
            return flower_groups, CutType.NONE

    def _handle_base_cases_filtered(self, log: List[Tuple[str]]) -> Tuple[List[Set[str]], CutType]:
        """
        Apply base case logic to the current sublog with frequency filtering.

        Parameters:
            log: sublog as subset of the original event log

        Returns:
            Tuple of groups and operator
        """
        # filter single activities
        log = self._single_activity_filtering(log)
        # filter empty traces
        log = self._empty_trace_filtering(log)
        # Apply the base case logic on the filtered log
        return super()._handle_base_cases(log)

    def _single_activity_filtering(self, log: List[Tuple[str]]) -> List[Tuple[str]]:
        """
        Filter out single activity traces that are not frequent enough.

        Parameters:
            log: sublog as subset of the original event log

        Returns:
            Filtered log
        """
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
        """
        Filter out empty traces that are not frequent enough.

        Parameters:
            log: sublog as subset of the original event log

        Returns:
            Filtered log
        """
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

    def xor_split_filtered(self, log: List[Tuple[str]], groups: List[Set[str]]) -> List[List[Tuple[str, ...]]]:
        """
        Filters the sublog before splitting logs based on xor operator. Figures out which activities should be
        handled as infrequent behavior based on affiliation to xor cut groups.

        Parameters:
            log: sublog as subset of event log
            groups: result groups of operator cuts to alphabet of corresponding log
        """
        for i, trace in enumerate(log.copy()):
            affiliation, removals = [], []

            for group in groups:
                trace_set = set(trace)
                group = set(group)

                # Continue with next trace, when group activities and trace activities are equal -> no filter needed
                if trace_set == group:
                    break
                # Continue with next group, when group and trace don't share any elements -> wrong cut group
                elif trace_set.isdisjoint(group):
                    continue
                # When no distinct group was found, probably infrequent behaviour, therefore find all candidates
                else:
                    if group.issubset(trace_set):
                        affiliation.append(group)

            # Proceed based on amounts of found affiliates
            if not affiliation:
                continue
            # No real case, when activity in trace that is not in split, put unknown to removals
            elif len(affiliation) == 1:
                removals = set(trace).difference(affiliation[0])
            # When multiple affiliates were found, discover actual affiliation by amount of
            elif len(affiliation) > 1:
                # get activity frequency in trace
                counts = defaultdict(int)
                for activity in trace:
                    counts[activity] += 1

                # sum up frequencies of all activities of an affiliate
                sums = defaultdict(int)
                for index, affiliate in enumerate(affiliation):
                    for activity in affiliate:
                        for k, freq in counts.items():
                            if activity == k:
                                sums[index] += freq

                # get affiliate with the highest frequency sum put all others to removals
                actual_affiliation = affiliation[max(sums, key=sums.get)]
                affiliation.remove(actual_affiliation)
                removals = set().union(*affiliation)

            # replace the old trace with a filtered trace
            new_trace = tuple([element for element in trace if element not in removals])
            log[i] = new_trace

        # Actual xor split call
        return super()._xor_split(log=log, cut=groups)

    def sequence_split_filtered(self, log: List[Tuple[str]], groups: List[Set[str]]) -> List[List[Tuple[str, ...]]]:
        """
        Filters the sublog before splitting logs based on sequence operator.

        Parameters:
            log: sublog as subset of event log
            groups: result groups of operator cuts to alphabet of corresponding log
        """
        def actual_order(res):
            """ Helper function to get the order of the activity sets in the result """
            return [set(x["activity"]) for x in res.values()]

        def sequence_condition_met(first: List[Set], second: List[Set]):
            """ Helper function to check if the first set list equals second set list"""
            # TODO [{"A"}, {"B"}, {"C"}, {"C"}, {"C"}] should become [{"A"}, {"B"}, {"C"}]
            # merge must be registered in result dict by accumulating the length of all merged

            return first == second

        def powerset(iterable):
            """ Helper function to get the powerset of a given list of removals"""
            s = list(iterable)
            return filter(lambda x: x != (), chain.from_iterable(combinations(s, r) for r in range(len(s) + 1)))

        for i, trace in enumerate(log.copy()):
            result = {}
            index = 0
            current_group = []

            for j, activity in enumerate(trace):
                # If no current group, start a new group with current activity
                if not current_group:
                    current_group = {"activity": activity, "length": 1}
                # If activity is the same as the current group, increment the length of the group
                elif activity == current_group["activity"]:
                    current_group["length"] += 1
                # If activity is different from the current group, append current group to result and start a new group
                else:
                    result[index] = current_group
                    index += 1
                    current_group = {"activity": activity, "length": 1}
            # Append the last group when finished iterating over the trace
            if current_group:
                result[index] = current_group

            current_max_size = 1
            total_max_size = max([v["length"] for k, v in result.items()])
            max_costs = None
            found_removals = []

            # Iterate over the trace and remove activities until the sequence condition is met
            while (not sequence_condition_met(actual_order(result), groups)
                   and current_max_size <= total_max_size):
                # If the current max size is smaller than the total max size times the threshold, break
                # Used to throttle due to the powerset calculation in case of large traces # TODO
                if current_max_size < total_max_size * self.threshold and total_max_size * self.threshold > 2:
                    break

                # Get Indices of all groups that are smaller or equal to the current max size and calculate the powerset
                removal_indices = [k for k, v in result.items() if current_max_size >= v["length"]]
                powerset_values = powerset(removal_indices)

                for removals in powerset_values:
                    new_res = result.copy()
                    costs = 0
                    # Calculate the costs of the removals and remove activity set from new result
                    for rm_index in removals:
                        costs += new_res[rm_index]["length"]
                        new_res.pop(rm_index, None)

                    # If the sequence condition is met, break
                    if sequence_condition_met(actual_order(new_res), groups):
                        max_costs = costs
                        total_max_size = max_costs
                        found_removals = removals
                        break

                if max_costs:
                    break

                # If no removal could fulfill the sequence condition
                current_max_size += 1

            # TODO minimize max_costs
            # if found, set new max list size to min(current costs, last max size)
            # edge case, costs 3 from 3 single lists, new max list size 3,
            # only create powerset that is not exceeding max costs
            # Since starting from 1, the first found removals should be quite good already

            # TODO handle cut groups len > 2
            # Not handled yet, must ignore activities in removals from groups > 1,
            # since they are handled by other operator

            # Remove all found removals from the trace
            for j in found_removals:
                del result[j]

            # Reconstruct the trace based on the filtered result
            trace = tuple([activity for k, v in result.items() for activity in [v["activity"]] * v["length"]])
            log[i] = trace

        return super()._sequence_split(log=log, cut=groups)

    def loop_split_filtered(self, log: List[Tuple[str]], groups: List[Set[str]]) -> List[List[Tuple[str, ...]]]:
        return super()._loop_split(log=log, cut=groups)
