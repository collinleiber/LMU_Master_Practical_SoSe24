import copy
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

        if empty_traces:
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
            # Find for each node the maximum frequency of outgoing edges
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
            return super()._projection_split(log, groups)
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
            """
            Helper function to check if the first set list equals second set list.
            sets of size > 1 are merged to one set before comparison.
            """
            second_not_base = [set(x) for x in second if len(x) > 1]

            current_group, last_group = None, None
            merged_first = []

            for activity in first:
                found = False
                for idx, not_base_set in enumerate(second_not_base):
                    if activity.issubset(not_base_set):
                        found = True
                        current_group = idx
                        break
                if found and (last_group is None or current_group != last_group):
                    merged_first.append(activity)
                    last_group = current_group
                elif found and current_group == last_group:
                    merged_first[-1] |= activity
                else:
                    merged_first.append(activity)
                    last_group = None

            # return if merged first set is equal to the second set or if all first sets are subsets of the second sets
            return (merged_first == second or
                    len(merged_first) == len(second) and all(f.issubset(s) for f, s in zip(merged_first, second)))

        def powerset(iterable):
            """ Helper function to get the powerset of a given list of removals"""
            s = list(iterable)
            return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))

        def merge_dict(d):
            """ Helper function to merge activities with the same activity in the result dict """
            keys = list(d.keys())
            idx = 0
            while idx < len(keys) - 1:
                current_key = keys[idx]
                next_key = keys[idx + 1]
                if d[current_key]['activity'] == d[next_key]['activity']:
                    d[current_key]['length'] += d[next_key]['length']
                    del d[next_key]
                    keys.remove(next_key)
                else:
                    idx += 1
            return d

        def get_removal_indices():
            removal_indices = []
            for trace_nr, values in sub_traces.items():
                groups_with_multiple_items = [g for g in groups if len(g) > 1]

                if groups_with_multiple_items:
                    for group in groups_with_multiple_items:
                        if values["activity"] not in group:
                            if current_max_size >= values["length"]:
                                removal_indices.append(trace_nr)
                            break

                        # Find outliers of groups with multiple items. TODO calc once
                        found = False
                        distance = 2
                        for x in range(max(0, trace_nr - distance),
                                       min(len(sub_traces), trace_nr + distance)):
                            if x != trace_nr and sub_traces[x]["activity"] in group:
                                found = True
                                break

                        if not found:
                            removal_indices.append(trace_nr)
                            break
                else:
                    if current_max_size >= values["length"]:
                        removal_indices.append(trace_nr)

            return removal_indices

        def build_sub_traces(_trace: Tuple[str]):
            """ Helper function to build sub traces with each sub trace containing only one activity """
            res = {}
            index = 0
            current_group = []

            for j, activity in enumerate(_trace):
                # If no current group, start a new group with current activity
                if not current_group:
                    current_group = {"activity": activity, "length": 1}
                # If activity is the same as the current group, increment the length of the group
                elif activity == current_group["activity"]:
                    current_group["length"] += 1
                # If activity is different from the current group, append current group to result and start a new group
                else:
                    res[index] = current_group
                    index += 1
                    current_group = {"activity": activity, "length": 1}
            # Append the last group when finished iterating over the trace
            if current_group:
                res[index] = current_group
            return res

        for i, trace in enumerate(log.copy()):
            # Build sub traces with each sub trace containing only one activity
            sub_traces = build_sub_traces(trace)

            total_max_size = max([v["length"] for k, v in sub_traces.items()])
            threshold_value = total_max_size * self.threshold
            current_max_size = 1

            max_costs, best_removal, visited = None, [], set()

            # If the sequence condition is already met, continue with the next trace
            trace_order = actual_order(sub_traces)
            if sequence_condition_met(trace_order, groups):
                continue

            # Iterate over the trace and find and minimize costs for removals that fulfill the sequence condition
            while not max_costs or (current_max_size < max_costs and current_max_size <= total_max_size):
                # If the current max size is smaller than the total max size times the threshold, break
                # Used to throttle due to the powerset calculation in case of large traces
                if current_max_size <= max(threshold_value, 3) and best_removal:
                    break

                # Get Indices of all groups that are smaller or equal to the current max size and calculate the powerset
                removal_indices = get_removal_indices()
                powerset_values = powerset(removal_indices)

                for removals in powerset_values:
                    if removals in visited:
                        continue
                    visited.add(removals)
                    copied_subs = copy.deepcopy(sub_traces)
                    costs = 0

                    # Calculate the costs of the removals and remove activity set from new result
                    for rm_index in removals:
                        costs += copied_subs[rm_index]["length"]
                        copied_subs.pop(rm_index, None)
                    if max_costs and costs >= max_costs:
                        break

                    # Merge activities with the same activity in the result dict
                    merged_subs = merge_dict(copied_subs)

                    # If the sequence condition is met, break
                    if sequence_condition_met(actual_order(merged_subs), groups):
                        if not max_costs or costs < max_costs:
                            max_costs = costs
                            best_removal = merged_subs.copy()

                # If no removal could fulfill the sequence condition
                if current_max_size <= total_max_size:
                    current_max_size += 1

            # Reconstruct the trace based on the filtered result
            trace = tuple([activity for k, v in best_removal.items()
                           for activity in [v["activity"]] * v["length"]])
            log[i] = trace
        return super()._projection_split(log=log, cut=groups)

    def loop_split_filtered(self, log: List[Tuple[str]], groups: List[Set[str]]) -> List[List[Tuple[str, ...]]]:
        """
        Filters the sublog before splitting logs based on loop operator.

        Parameters:
            log: sublog as subset of event log
            groups: result groups of operator cuts to alphabet of corresponding log
        """
        loop_body, redo = groups[0], groups[1:]
        removals = defaultdict(set)  # Could be replaced by simple counter, but this way more comprehensible

        for x, trace in enumerate(log):
            found = set()
            # Find infrequent activities at trace start
            for i, activity in enumerate(trace):
                if activity in loop_body:
                    # TODO Questionable Condition
                    # Should already found activities be removed when other activities from loop body lacking?
                    if activity in found:
                        removals[x].add(i)
                    found.add(activity)
                    if found == loop_body:
                        break
                else:
                    removals[x].add(i)
            found.clear()
            # Find infrequent activities at trace end
            for i, activity in enumerate(reversed(trace)):
                if activity in loop_body:
                    if activity in found:
                        removals[x].add(len(trace) - i - 1)
                    found.add(activity)
                    if found == loop_body:
                        break
                else:
                    removals[x].add(len(trace) - i - 1)

        base_result = super()._loop_split(log=log, cut=groups)
        # Add amount of empty traces based on removals
        base_result[0] = base_result[0] + [('',)] * len(list(chain(*removals.values())))
        return base_result
