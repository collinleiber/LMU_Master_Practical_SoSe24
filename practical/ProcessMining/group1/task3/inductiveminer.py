from typing import List, Tuple, Dict, Set
import pm4py
from practical.ProcessMining.group1.shared.utils import read_txt_test_logs, event_log_to_dataframe


class InductiveMiner:
    def __init__(self, log: List[Tuple] = None):
        self.log = log
        self.alphabet = self._get_alphabet(self.log)
        self.dfg, self.start_activities, self.end_activities = self._get_dfg(self.log)

    def run(self):
        sequence_cut = self._sequence_cut(self.dfg, self.start_activities, self.end_activities)
        xor_cut = self._xor_cut(self.dfg, self.start_activities, self.end_activities)
        parallel_cut = self._parallel_cut(self.dfg, self.start_activities, self.end_activities)
        loop_cut = self._loop_cut(self.dfg, self.start_activities, self.end_activities)

        if self._is_nontrivial(sequence_cut):
            self.log = self._split_log(self.log, sequence_cut)
        elif self._is_nontrivial(xor_cut):
            self.log = self._split_log(self.log, xor_cut)
        elif self._is_nontrivial(parallel_cut):
            self.log = self._split_log(self.log, parallel_cut)
        elif self._is_nontrivial(loop_cut):
            self.log = self._split_log(self.log, loop_cut)
        else:  # flower model
            flower_groups = [{activity} for activity in sorted(list(self.alphabet))]
            self.log = self._split_log(self.log, flower_groups)
            pass

    def _is_nontrivial(self, max_groups: List[Set[str]]) -> bool:
        return len(max_groups) > 1 if max_groups else False

    def _get_alphabet(self, log: List[Tuple]) -> Set[str]:
        return set([activity for trace in log for activity in trace])

    def _get_dfg(self, log: List[Tuple]) -> Tuple[Dict[Tuple[str, str], int], Dict[str, int], Dict[str, int]]:
        # TODO: Own implementation of dfg
        return pm4py.discover_dfg(pm4py.format_dataframe(event_log_to_dataframe(log), case_id='case_id',
                                                         activity_key='activity', timestamp_key='timestamp'))

    def _sequence_cut(self, dfg: Dict[Tuple[str, str], int], start: Dict[str, int],
                      end: Dict[str, int]) -> List[Set[str]]:
        pass

    def _xor_cut(self, dfg: Dict[Tuple[str, str], int], start: Dict[str, int], end: Dict[str, int]) -> List[Set[str]]:
        pass

    def _parallel_cut(self, dfg: Dict[Tuple[str, str], int], start: Dict[str, int],
                      end: Dict[str, int]) -> List[Set[str]]:
        # For pseudo code: see
        # Robust Process Mining with Guarantees (https://link.springer.com/book/10.1007/978-3-030-96655-3)
        edges = dfg.keys()
        start_activities = set(start.keys())
        end_activities = set(end.keys())
        groups = [{activity} for activity in sorted(list(self.alphabet))]

        # Merge groups that are connected by edges
        done = False
        while not done:
            done = True
            for i, group in enumerate(groups[:-1]):
                other_group = groups[i + 1]
                # Check if there are no edges between the two groups
                if any((a, b) not in edges or (b, a) not in edges for a in group for b in other_group):
                    # Merge groups
                    groups[i] = group.union(other_group)
                    groups.pop(i + 1)
                    done = False
                    break

        # Filter out groups that do not contain start and end activities
        groups = sorted((group for group in groups if group & start_activities and group & end_activities), key=len)
        return groups

    def _loop_cut(self, dfg: Dict[Tuple[str, str], int], start: Dict[str, int], end: Dict[str, int]) -> List[Set[str]]:
        edges = dfg.keys()
        start_activities = set(start.keys())
        end_activities = set(end.keys())
        # For pseudo code: see
        # Robust Process Mining with Guarantees (https://link.springer.com/book/10.1007/978-3-030-96655-3)

        # Merge start and end activities into do group
        groups = [start_activities.union(end_activities)]

        # Remove start and end activities from the dfg and add as loop group
        inner_edges = [edge for edge in edges if edge[0] not in groups[0] and edge[1] not in groups[0]]
        groups.append(set([activity for edge in inner_edges for activity in edge]))

        # Exclude sets that are non-reachable from start/end activities from the loop groups
        pure_start_activities = start_activities.difference(end_activities)  # only start, not end activity
        # Put all activities in the do-group that follow a start activity which is not an end activity
        # (a loop activity can only follow a start activity if it is also an end activity)
        for a in pure_start_activities:
            for (x, b) in edges:
                if x == a:
                    group_a, group_b = set(), set()
                    for group in groups:
                        group_a = group if a in group else group_a
                        group_b = group if b in group else group_b
                    groups = [group for group in groups if group != group_a and group != group_b]
                    groups.insert(0, group_a.union(group_b))

        pure_end_activities = end_activities.difference(start_activities)  # only end, not start activity
        # Put all activities in the do-group that precede an end activity which is not a start activity
        # (a loop activity can only precede an end activity if it is also a start activity)
        for b in pure_end_activities:
            for (a, x) in edges:
                if x == b:
                    group_a, group_b = set(), set()
                    for group in groups:
                        if a in group:
                            group_a = group
                        if b in group:
                            group_b = group
                    groups = [group for group in groups if group not in [group_a, group_b]]
                    groups.insert(0, group_a.union(group_b))
        # Check start completeness
        i = 1
        while i < len(groups):
            merge = False
            for a in groups[i]:
                for (x, b) in edges:
                    if x == a and b in start_activities:
                        if not any((a, s) in edges for s in start_activities):
                            merge = True
            if merge:
                groups[0] = groups[0].union(groups[i])
                groups.pop(i)
            else:
                i += 1
        # Check end completeness
        i = 1
        while i < len(groups):
            merge = False
            for a in groups[i]:
                for (b, x) in edges:
                    if x == a and b in end_activities:
                        if not any((e, a) in edges for e in end_activities):
                            merge = True
            if merge:
                groups[0] = groups[0].union(groups[i])
                groups.pop(i)
            else:
                i += 1

        # return cut if more than one group (i.e. do- and loop-group found)
        return groups if len(groups) > 1 else []

    def _split_log(self, log: List[Tuple[str]], cut: List[Set[str]]) -> List[Tuple[str]]:
        new_traces = [''.join(map(str, group)) for group in cut]
        new_log = []
        for trace in log:
            trace = ''.join(map(str, trace))
            for new_trace in new_traces:
                while True:
                    sublog = self._find_subsequence_in_arbitrary_order(trace, new_trace)
                    if len(sublog) > 0:
                        new_log.append(tuple(sublog))
                        trace = trace.replace(sublog, '_', 1)  # TODO: breaks if activity name includes '_'
                    else:
                        break
        return new_log

    def _find_subsequence_in_arbitrary_order(self, main: str, sub: str) -> str:
        sub_len = len(sub)
        sorted_sub = sorted(sub)

        for i in range(len(main) - sub_len + 1):
            window = main[i:i + sub_len]
            if sorted(window) == sorted_sub:
                return window
        return ''
