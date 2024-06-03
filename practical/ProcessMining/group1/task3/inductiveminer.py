from enum import Enum
from typing import List, Tuple, Dict, Set
import pm4py
import re
from practical.ProcessMining.group1.shared.utils import event_log_to_dataframe


class CutType(Enum):
    SEQUENCE = '→'
    XOR = '×'
    PARALLEL = '∧'
    LOOP = '↺'
    NONE = ''


class InductiveMiner:
    TAU = '𝜏'

    def __init__(self, log: List[Tuple] = None):
        self.logs = log
        self.alphabet = self._get_alphabet(self.logs)
        self.dfg, self.start_activities, self.end_activities = self._get_dfg(self.logs)
        self.process_tree_str = '()'

    def print_process_tree(self):
        print(self.process_tree_str)

    def run(self):
        sublogs = [self.logs]
        i = 0
        while i < len(sublogs):
            log = sublogs[i]
            # update dfg, start_activities, end_activities
            dfg, start_activities, end_activities = self._get_dfg(log)

            # check for base cases
            base_cut, operator = self._handle_base_cases(log)
            if base_cut:
                self._build_process_tree(base_cut, operator)
                sublogs.pop(i)
                continue

            # cut selection and log splitting
            sequence_cut = self._sequence_cut(dfg, start_activities, end_activities)
            xor_cut = self._xor_cut(dfg, start_activities, end_activities)
            parallel_cut = self._parallel_cut(dfg, start_activities, end_activities)
            loop_cut = self._loop_cut(dfg, start_activities, end_activities)

            new_sublogs = None
            if self._is_nontrivial(sequence_cut):
                new_sublogs = self._split_log(log, sequence_cut)
                self._build_process_tree(sequence_cut, CutType.SEQUENCE)
            elif self._is_nontrivial(xor_cut):
                new_sublogs = self._split_log(log, xor_cut)
                self._build_process_tree(xor_cut, CutType.XOR)
            elif self._is_nontrivial(parallel_cut):
                new_sublogs = self._split_log(log, parallel_cut)
                self._build_process_tree(parallel_cut, CutType.PARALLEL)
            elif self._is_nontrivial(loop_cut):
                new_sublogs = self._split_log(log, loop_cut)
                self._build_process_tree(loop_cut, CutType.LOOP)
            else:  # fall through - flower model
                flower_groups = self._handle_fall_through(log)
                new_sublogs = self._split_log(log, flower_groups)
                self._build_process_tree(flower_groups, CutType.LOOP)
                break  # flower model is the last cut?

            if new_sublogs:
                sublogs.pop(i)
                sublogs.extend(new_sublogs)
            else:
                i += 1

        self._clean_process_tree()
        return sublogs

    def _build_process_tree(self, groups: List[Set[str]], cut_type: CutType = None, ) -> str:
        tree = self.process_tree_str
        group_str = ', '.join([', '.join(activity) for group in groups for activity in group])
        cut_str = f'{cut_type.value}' if cut_type != CutType.NONE else ''
        if len(group_str) > 1:
            new_cut_str = f'{cut_str}({group_str})'
        else:
            new_cut_str = f'{cut_str}{group_str}'
        if tree == '()':
            tree = new_cut_str
        else:
            match = self._find_subsequence_in_arbitrary_order(tree, group_str.replace(f', {self.TAU}', ''))
            tree = tree.replace(match, f'{new_cut_str}')
        self.process_tree_str = tree
        return self.process_tree_str

    def _clean_process_tree(self):
        self.process_tree_str = self.process_tree_str.replace(', ()', '')
        return self.process_tree_str

    def _is_nontrivial(self, max_groups: List[Set[str]]) -> bool:
        return len(max_groups) > 1 if max_groups else False

    def _get_alphabet(self, log: List[Tuple]) -> Set[str]:
        return set([activity for trace in log for activity in trace])
    def _get_alphabet_from_dfg(self, dfg: Dict[Tuple[str, str], int]) -> Set[str]:
        return set([activity for edge in dfg.keys() for activity in edge])

    def _get_dfg(self, log: List[Tuple]) -> Tuple[Dict[Tuple[str, str], int], Dict[str, int], Dict[str, int]]:
        # TODO: Own implementation of dfg
        return pm4py.discover_dfg(pm4py.format_dataframe(event_log_to_dataframe(log), case_id='case_id',
                                                         activity_key='activity', timestamp_key='timestamp'))

    def _handle_base_cases(self, log: List[Tuple[str]]) -> Tuple[List[Set[str]], CutType]:
        traces = [''.join(map(str, trace)) for trace in log]
        alphabet = [a for a in self._get_alphabet(log) if a != '']
        base_activity = set(alphabet[0]) if alphabet else set()
        tau_activity = set(self.TAU)
        operator = CutType.NONE
        groups = []

        if len(alphabet) == 1:
            if all(len(trace) == 1 for trace in traces):  # exactly once (1)
                operator = CutType.NONE
                groups = [base_activity]
            elif all(len(trace) <= 1 for trace in traces):  # never or once (0,1)
                operator = CutType.XOR
                groups += [base_activity, tau_activity]
            elif all(len(trace) > 0 for trace in traces):  # once or many times (1..*)
                operator = CutType.LOOP
                groups += [base_activity, tau_activity]
            else:  # never, once or many times (0..*)
                operator = CutType.LOOP
                groups += [tau_activity, base_activity]

        return groups, operator

    def _handle_fall_through(self, log: List[Tuple[str]]) -> List[Set[str]]:
        flower_groups = [set(activity) for activity in sorted(list(self._get_alphabet(log)))]
        flower_groups.append(set(self.TAU))
        return flower_groups

    def _sequence_cut(self, dfg: Dict[Tuple[str, str], int], start: Dict[str, int], end: Dict[str, int]) -> List[
        Set[str]]:
        alphabet = self._get_alphabet_from_dfg(dfg)
        if not alphabet:
            return []

        start_activities = set(start.keys())
        end_activities = set(end.keys())

        partitions = []

        all_activities = set()
        for (a, b) in dfg:
            all_activities.add(a)
            all_activities.add(b)

        first_partition = start_activities
        partitions.append(first_partition)

        remaining_activities = all_activities - start_activities - end_activities
        middle_partition = remaining_activities
        final_partition = end_activities

        if middle_partition:
            partitions.append(middle_partition)

        partitions.append(final_partition)
        return partitions

    def _xor_cut(self, dfg: Dict[Tuple[str, str], int], start: Dict[str, int], end: Dict[str, int]) -> List[Set[str]]:
        partitions = []

        all_activities = set()
        for (a, b) in dfg:
            all_activities.add(a)
            all_activities.add(b)

        components = []
        visited = set()

        def dfs(activity, component):
            stack = [activity]
            while stack:
                node = stack.pop()
                if node not in visited:
                    visited.add(node)
                    component.add(node)
                    for successor in [b for (a, b) in dfg if a == node]:
                        stack.append(successor)
                    for predecessor in [a for (a, b) in dfg if b == node]:
                        stack.append(predecessor)

        for activity in all_activities:
            if activity not in visited:
                component = set()
                dfs(activity, component)
                components.append(component)

        for component in components:
            partitions.append(component)

        return partitions

    def _parallel_cut(self, dfg: Dict[Tuple[str, str], int], start: Dict[str, int],
                      end: Dict[str, int]) -> List[Set[str]]:
        # For pseudo code: see
        # Robust Process Mining with Guarantees (https://link.springer.com/book/10.1007/978-3-030-96655-3)
        edges = dfg.keys()
        start_activities = set(start.keys())
        end_activities = set(end.keys())
        groups = [{activity} for activity in set([activity for edge in edges for activity in edge])]

        # Merge groups that are connected by edges
        done = False
        while not done:
            done = True
            i = 0
            while i < len(groups):
                j = i + 1
                while j < len(groups):
                    group_a, group_b = groups[i], groups[j]
                    if any((a, b) not in edges or (b, a) not in edges for a in group_a for b in group_b):
                        groups[i] = group_a.union(group_b)
                        groups.pop(j)
                        done = False
                    else:
                        j += 1
                if not done:
                    break
                i += 1

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
        groups = [group for group in groups if group != set()]
        return groups if len(groups) > 1 else []

    def _split_log(self, log: List[Tuple[str]], cut: List[Set[str]]) -> List[List[Tuple[str]]]:
        # projective splitting is used for sequence / parallel cuts and fall through
        new_traces = [''.join(map(str, group)).replace(self.TAU, '') for group in cut]
        old_traces = [''.join(map(str, trace)) for trace in log]
        new_log = []
        for new_trace in new_traces:
            sub_log = []
            for i, old_trace in enumerate(old_traces):
                while True:
                    sub_trace = self._find_subsequence_in_arbitrary_order(old_trace, new_trace)
                    if len(sub_trace) > 0:
                        sub_log.append(tuple(sub_trace))
                        old_trace = old_trace.replace(sub_trace, '_', 1)  # TODO: breaks if activity name includes '_'
                    else:
                        break
            new_log.append(sub_log)
        return new_log

    def _find_subsequence_in_arbitrary_order(self, main: str, sub: str) -> str:
        sub_len = len(sub)
        sorted_sub = sorted(sub)

        for i in range(len(main) - sub_len + 1):
            window = main[i:i + sub_len]
            if sorted(window) == sorted_sub:
                return window
        return ''
