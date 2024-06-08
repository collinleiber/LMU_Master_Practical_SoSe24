from enum import Enum
from typing import List, Tuple, Dict, Set, Optional
import pm4py
import pandas as pd
from practical.ProcessMining.group1.shared.utils import event_log_to_dataframe
from pm4py.objects.process_tree.obj import ProcessTree
from pm4py.visualization.process_tree import visualizer as pt_visualizer
from pm4py.objects.conversion.log import converter as log_converter


class CutType(Enum):
    """
    Enum class to represent the different types of cuts that can be applied to the directly-follows graph.
    """
    SEQUENCE = '→'
    XOR = '×'
    PARALLEL = '∧'
    LOOP = '↺'
    NONE = ''


class InductiveMiner:
    """
    Inductive Miner implementation based on the paper:
    "Process Mining: Discovery, Conformance and Enhancement of Business Processes" by Wil M.P. van der Aalst

    Attributes:
        event_log: List of traces
        alphabet: Set of unique activities in the initial event log
        dfg: Directly-follows graph for the initial event log
        start_activities: Start activities in the log
        end_activities: End activities in the log
        process_tree_str: String representation of the process tree
    """
    TAU = '𝜏'

    def __init__(self, event_log: Optional[List[Tuple[str]]] = None):
        """
        Initialize the Inductive Miner with an event log.

        Parameters:
            event_log: List of traces
        """
        self.event_log = event_log
        self.alphabet = self._get_alphabet(self.event_log)
        self.dfg, self.start_activities, self.end_activities = self._get_dfg(self.event_log)
        self.process_tree_str = '()'  # start with an empty process tree

    def run(self) -> None:
        """
        Main method to run the Inductive Miner algorithm. It iteratively applies different types of cuts
        (sequence, XOR, parallel, loop) to the dfg, splits the event log into sublogs, and builds up the
        process tree accordingly.
        """
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
                    new_sublogs = self._split_log(log, groups, operator)
                    sublogs.extend(new_sublogs)
                    # Build the corresponding part of the process tree
                    self._build_process_tree(groups, operator)
                else:  # If fall through case, build the flower model
                    self._build_process_tree(groups, CutType.LOOP)

            # Remove the old sublog from the list
            sublogs.remove(log)

    def _apply_cut(self, log: List[Tuple[str]], dfg: Dict[Tuple[str, str], int], start_activities: Dict[str, int],
                   end_activities: Dict[str, int]) -> Tuple[List[Set[str]], CutType]:
        """
        Applies different types of cuts to the current sublog and builds the corresponding part of the process tree.

        Parameters:
            log: List of traces
            dfg: Directly-follows graph
            start_activities: Start activities in the log
            end_activities: End activities in the log

        Returns:
            List of new sublogs resulting from the split. Empty list if no operator could be applied.
        """
        # Try to apply different types of cuts to the current sublog
        sequence_cut = self._sequence_cut(dfg, start_activities, end_activities)
        xor_cut = self._xor_cut(dfg, start_activities, end_activities)
        parallel_cut = self._parallel_cut(dfg, start_activities, end_activities)
        loop_cut = self._loop_cut(dfg, start_activities, end_activities)

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

    def print_process_tree(self) -> None:
        """
        Prints the string representation of the process tree.
        """
        print(self.process_tree_str)

    def _build_process_tree(self, groups: List[Set[str]], cut_type: Optional[CutType] = None) -> str:
        """
        Builds the process tree based on the groups and cut type provided.

        Parameters:
            groups: List of groups of activities that form the process tree.
            cut_type: The type of cut (SEQUENCE, XOR, PARALLEL, LOOP, NONE) that was applied to form the groups.

        Returns:
            The updated string representation of the process tree.
        """
        # Get the current process tree
        tree = self.process_tree_str

        # Convert the groups into a string
        group_str = ', '.join([', '.join(activity) for group in groups for activity in group])

        # Get the string representation of the cut type
        cut_str = f'{cut_type.value}' if cut_type != CutType.NONE else ''

        # If the group has more than one activity, wrap it in parentheses and prepend the cut type
        if len(group_str) > 1:
            new_cut_str = f'{cut_str}({group_str})'
        else:  # If the group has only one activity, prepend the cut type
            new_cut_str = f'{cut_str}{group_str}'

        # If the current process tree is empty (no cut applied yet), replace it with the new cut string
        if tree == '()':
            tree = new_cut_str
        else:
            # If the current process tree is not empty, find the subsequence in the tree that matches the group string
            # and replace it with the new cut string
            match = self._find_subsequence_in_arbitrary_order(tree, group_str.replace(f'{self.TAU}, ', ''))
            tree = tree.replace(match, f'{new_cut_str}')

        # Update the process tree
        self.process_tree_str = tree

        return self.process_tree_str

    def _is_nontrivial(self, max_groups: Optional[List[Set[str]]]) -> bool:
        """
        Checks if the number of groups is greater than 1 (i.e. the cut would separate the dfg further).

        Parameters:
            max_groups: List of groups of activities that form the cut.

        Returns:
            True if the cut is nontrivial, False otherwise.
        """
        return len(max_groups) > 1 if max_groups else False

    def _get_alphabet(self, log: List[Tuple[str]]) -> Set[str]:
        """
        Extracts the unique activities from the event log.

        Parameters:
            log: List of traces

        Returns:
            Set of unique activities in the event log.
        """
        return {activity for trace in log for activity in trace}

    def _get_alphabet_from_dfg(self, dfg: Dict[Tuple[str, str], int]) -> Set[str]:
        """
        Extracts the unique activities from the directly-follows graph.

        Parameters:
            dfg: Directly-follows graph

        Returns:
            Set of unique activities in the dfg.
        """
        return {activity for edge in dfg.keys() for activity in edge}

    def _get_dfg(self, log: List[Tuple]) -> Tuple[Dict[Tuple[str, str], int], Dict[str, int], Dict[str, int]]:
        """
        Builds the directly-follows graph (dfg) from the event log and extracts the start and end activities.

        Parameters:
            log: List of traces

        Returns:
            Tuple containing the dfg, start activities, and end activities.
        """
        dfg = {}
        start_activities = {}
        end_activities = {}

        for trace in log:
            if trace[0] in start_activities:
                start_activities[trace[0]] += 1
            else:
                start_activities[trace[0]] = 1

            if trace[-1] in end_activities:
                end_activities[trace[-1]] += 1
            else:
                end_activities[trace[-1]] = 1

            for i in range(len(trace) - 1):
                pair = (trace[i], trace[i + 1])
                if pair in dfg:
                    dfg[pair] += 1
                else:
                    dfg[pair] = 1

        return dfg, start_activities, end_activities
        # return pm4py.discover_dfg(pm4py.format_dataframe(event_log_to_dataframe(log), case_id='case_id',
        #                                                  activity_key='activity', timestamp_key='timestamp'))

    def _handle_base_cases(self, log: List[Tuple[str]]) -> Tuple[List[Set[str]], CutType]:
        """
        Handles the base cases (i.e. only one type of activity in the log) for the Inductive Miner algorithm.

        Parameters:
            log: List of traces

        Returns:
            Tuple containing the groups of activities that form the base case and the corresponding cut type.
        """
        # Convert traces to strings for easier comparison
        traces = [''.join(map(str, trace)) for trace in log]
        # Extract unique activities from the log, excluding the empty traces
        alphabet = [a for a in self._get_alphabet(log) if a != '']
        # Initialize base activity
        base_activity = set(alphabet[0]) if alphabet else set()
        tau_activity = set(self.TAU)
        operator = CutType.NONE
        groups = []

        # If there is only one unique activity in the log
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
        """
        Handles the fall-through case (flower model) for the Inductive Miner algorithm.

        Parameters:
            log: List of traces

        Returns:
            List of groups of activities that form the flower model.
        """
        # Tau in the do part of the loop cut (= 0..* execution of any activity)
        flower_groups = [set(self.TAU), [set(activity) for activity in sorted(list(self._get_alphabet(log)))]]
        return flower_groups

    def _sequence_cut(self, dfg: Dict[Tuple[str, str], int], start: Dict[str, int],
                      end: Dict[str, int]) -> List[Set[str]]:
        """
        Applies the sequence cut to the directly-follows graph (dfg).

        Parameters:
            dfg: Directly-follows graph
            start: Start activities in the log
            end: End activities in the log

        Returns:
            List of groups of activities that form the sequence cut.
        """
        alphabet = self._get_alphabet_from_dfg(dfg)
        if not alphabet:
            return []

        # Initialize partitions with the start activities as the first group
        partitions = [set(start.keys())]
        remaining_activities = alphabet - partitions[0]

        # Function to check if we can add an activity to a partition
        def can_add_to_partition(activity, partition):
            for p in partition:
                if (activity, p) in dfg or (p, activity) not in dfg:
                    return False
            return True

        # Build partitions iteratively
        while remaining_activities:
            next_partition = set()
            for activity in sorted(remaining_activities):
                if can_add_to_partition(activity, partitions[-1]):
                    partitions[-1].add(activity)
                else:
                    next_partition.add(activity)
            if next_partition:
                partitions.append(next_partition)
            remaining_activities -= partitions[-1]

        # Validate the sequence cut against the given conditions
        for i in range(len(partitions) - 1):
            for j in range(i + 1, len(partitions)):
                for ai in partitions[i]:
                    for aj in partitions[j]:
                        if (aj, ai) in dfg or (ai, aj) not in dfg:
                            return []

        return partitions if len(partitions) > 1 else []

    def _xor_cut(self, dfg: Dict[Tuple[str, str], int], start: Dict[str, int], end: Dict[str, int]) -> List[Set[str]]:
        partitions = []

        # Collect all activities from the dfg
        all_activities = set()
        for (a, b) in dfg:
            all_activities.add(a)
            all_activities.add(b)
        all_activities.update(start.keys())
        all_activities.update(end.keys())

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
        """
        Applies the parallel cut to the directly-follows graph (dfg).

        Parameters:
            dfg: Directly-follows graph
            start: Start activities in the log
            end: End activities in the log

        Returns:
            List of groups of activities that form the parallel cut.
        """
        # Extract edges from the dfg
        edges = dfg.keys()
        start_activities = set(start.keys())
        end_activities = set(end.keys())
        # Initialize groups with individual activities
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
                    # If any pair of activities from the two groups is not connected by an edge in both directions,
                    # merge the two groups
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
        """
        Applies the loop cut to the directly-follows graph (dfg).

        Parameters:
            dfg: Directly-follows graph
            start: Start activities in the log
            end: End activities in the log

        Returns:
            List of groups of activities that form the loop cut.
        """
        # Extract edges from the dfg
        edges = dfg.keys()
        start_activities = set(start.keys())
        end_activities = set(end.keys())

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
        # all loop activities must be able to reach the start activities
        i = 1
        while i < len(groups):
            merge = False
            for a in groups[i]:
                for (x, b) in edges:
                    if x == a and b in start_activities:
                        if not any((a, s) in edges for s in start_activities):  # no direct edge from activity to start
                            merge = True  # merge with the do-group as it cannot be in the loop-group
            if merge:
                groups[0] = groups[0].union(groups[i])
                groups.pop(i)
            else:
                i += 1
        # Check end completeness
        # all loop activities must be able to be reached from the end activities
        i = 1
        while i < len(groups):
            merge = False
            for a in groups[i]:
                for (b, x) in edges:
                    if x == a and b in end_activities:
                        if not any((e, a) in edges for e in end_activities):  # no direct edge from end to activity
                            merge = True  # merge with the do-group as it cannot be in the loop-group
            if merge:
                groups[0] = groups[0].union(groups[i])
                groups.pop(i)
            else:
                i += 1

        # return cut if more than one group (i.e. do- and loop-group found)
        groups = [group for group in groups if group != set()]
        return groups if len(groups) > 1 else []

    def _split_log(self, log: List[Tuple[str]], cut: List[Set[str]], operator: CutType) -> List[List[Tuple[str]]]:
        if operator == CutType.SEQUENCE:
            return self._sequence_split(log, cut)
        elif operator == CutType.XOR:
            return self._xor_split(log, cut)
        elif operator == CutType.PARALLEL:
            return self._parallel_split(log, cut)
        elif operator == CutType.LOOP:
            return self._loop_split(log, cut)
        else:
            return []

    def _sequence_split(self, log: List[Tuple[str]], cut: List[Set[str]]) -> List[List[Tuple[str]]]:
        sublogs = [[] for _ in range(len(cut))]

        for trace in log:
            trace_list = list(trace)
            for i, group in enumerate(cut):
                sub_trace = tuple(activity for activity in trace_list if activity in group)
                if sub_trace:
                    sublogs[i].append(sub_trace)
                trace_list = [activity for activity in trace_list if activity not in group]

        sublogs = [sublog for sublog in sublogs if sublog]
        return sublogs

    def _xor_split(self, log: List[Tuple[str]], cut: List[Set[str]]) -> List[List[Tuple[str]]]:
        sublogs = [[] for _ in range(len(cut))]

        for trace in log:
            for i, group in enumerate(cut):
                sub_trace = tuple(activity for activity in trace if activity in group)
                if sub_trace:
                    sublogs[i].append(sub_trace)
                    break

        sublogs = [sublog for sublog in sublogs if sublog]
        return sublogs

    def _parallel_split(self, log: List[Tuple[str]], cut: List[Set[str]]) -> List[List[Tuple[str, ...]]]:
        """
        Splits the event log based on the groups provided.

        Parameters:
            log: List of traces
            cut: List of groups of activities that form the cut

        Returns:
            List of sublogs resulting from the parallel split.
        """
        sublogs = []
        # Iterate over the groups in the cut
        for partition in cut:
            sublog = []
            # Iterate over the traces in the log
            for trace in log:
                # Take activtites from the trace that are in the partition
                sub_trace = tuple(activity for activity in trace if activity in partition)
                if sub_trace:
                    sublog.append(sub_trace)
            sublogs.append(sorted(sublog))
        return sublogs

    def _loop_split(self, log: List[Tuple[str]], cut: List[Set[str]]) -> List[List[Tuple[str]]]:
        """
        Splits the event log based on the groups provided.

        Parameters:
            log: List of traces
            cut: List of groups of activities that form the cut

        Returns:
            List of sublogs resulting from the loop split.
        """
        # Convert the groups in the cut and the traces in the log to strings for easier comparison
        new_traces = [''.join(map(str, group)).replace(self.TAU, '') for group in cut]
        old_traces = [''.join(map(str, trace)) for trace in log]
        new_log = []

        # Iterate over the new traces (groups in the cut)
        for new_trace in new_traces:
            sub_log = []
            # Iterate over the old traces (traces in the log)
            for i, old_trace in enumerate(old_traces):
                # Find subsequences in the old trace that match the new trace and add them to the sublog
                while True:
                    sub_trace = self._find_subsequence_in_arbitrary_order(old_trace, new_trace)
                    if len(sub_trace) > 0:  # if a subsequence is found
                        sub_log.append(tuple(sub_trace))
                        # Remove the found subsequence from the old trace
                        old_trace = old_trace.replace(sub_trace, '_', 1)  # TODO: breaks if activity name includes '_'
                    else:
                        break
            # Add the sublog to the new log
            new_log.append(sub_log)

        return new_log

    def _find_subsequence_in_arbitrary_order(self, main: str, sub: str) -> str:
        """
        Finds a subsequence in the main string that contains the same characters as the sub string,
        regardless of their order.

        Parameters:
            main: The main string in which to find the subsequence.
            sub: The sub string whose characters should be found in the main string.

        Returns:
            The found subsequence in the main string. If no such subsequence is found, returns an empty string.
        """
        sub_len = len(sub)
        # Sort the characters in the sub string for comparison
        sorted_sub = sorted(sub)

        # Iterate over the main string with a sliding window
        for i in range(len(main) - sub_len + 1):
            # Get a window of characters in the main string of the same length as the sub string
            window = main[i:i + sub_len]
            # If the sorted characters in the window match the sorted characters in the sub string,
            # return the window as the found subsequence
            if sorted(window) == sorted_sub:
                return window
        # If no matching subsequence is found, return an empty string
        return ''

    def visualize_process_tree(self):
        """
        Visualizes the process tree using the specified symbols for SEQUENCE, XOR, PARALLEL, and LOOP cuts.

        Converts the event log to a dataframe, discovers the process tree using the inductive miner algorithm,
        replaces the operator labels with the corresponding symbols, and visualizes the process tree.

        The process tree is displayed using the specified format.
        """
        # Convert the event log to a dataframe for pm4py
        event_log_df = self._event_log_to_dataframe(self.event_log)
        log = log_converter.apply(event_log_df)
        tree = pm4py.discover_process_tree_inductive(log)

        def replace_labels(node):
            """
            Recursively replaces operator labels in the process tree nodes with the specified symbols.

            Parameters:
                node: The current node in the process tree.
            """
            if node.operator is not None:
                if node.operator == 'sequence':
                    node.operator = '→'
                elif node.operator == 'xor':
                    node.operator = '×'
                elif node.operator == 'parallel':
                    node.operator = '∧'
                elif node.operator == 'loop':
                    node.operator = '↺'

            # Recursively replace labels in child nodes
            for child in node.children:
                if child.children:
                    replace_labels(child)

        # Replace labels in the process tree
        replace_labels(tree)

        # Visualization settings
        parameters = {'format': 'png'}  # Can change later to other formats
        gviz = pt_visualizer.apply(tree, parameters=parameters)
        pt_visualizer.view(gviz)
        # optionally save the image if needed
        # pt_visualizer.save(gviz, "process_tree.png")

    def _event_log_to_dataframe(self, log):
        """
        Converts the event log to a pandas DataFrame formatted for pm4py.

        Parameters:
            log: List of traces

        Returns:
            A pandas DataFrame with the event log formatted for pm4py.
        """
        data = []
        for i, trace in enumerate(log):
            for event in trace:
                data.append({"case_id": i, "activity": event, "timestamp": i})
        return pm4py.format_dataframe(pd.DataFrame(data), case_id='case_id', activity_key='activity',
                                      timestamp_key='timestamp')
