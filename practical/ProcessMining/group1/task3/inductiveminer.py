import logging
from collections import defaultdict
from enum import Enum
from typing import List, Tuple, Dict, Set, Optional
import graphviz
import pandas as pd
from IPython.display import Image, display
import pm4py
from pm4py.visualization.process_tree import visualizer as pt_visualizer
from pm4py.objects.conversion.log import converter as log_converter
from practical.ProcessMining.group1.shared import utils
from pm4py.visualization.petri_net import visualizer as pn_vis
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.visualization.process_tree import visualizer as pt_vis
from pm4py.objects.conversion.process_tree import converter as pt_to_petri_converter

FILE_PATH_CSV = '../task2/sorted_session_data1.csv'
logging.basicConfig(level="INFO")  # Change to DEBUG for prints

parameters = {
    'noise_threshold': 0.2
}

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
        self.net, self.initial_marking, self.final_marking = None, None, None

    def __str__(self):
        return self.process_tree_str

    def run(self) -> None:
        """
        Main method to run the Inductive Miner algorithm. It iteratively applies different types of cuts
        (XOR, sequence, parallel, loop) to the dfg, splits the event log into sublogs, and builds up the
        process tree accordingly.
        """
        # Initialize the list of sublogs with the original event log
        sublogs = [self.event_log]

        # Iterate over the sublogs until the list is empty
        while len(sublogs) > 0:
            log = sublogs[0]
            result, groups, new_sublogs = self.recursion_step(log)

            # When no operator could be applied, return
            if not result:
                self._build_process_tree(groups, CutType.LOOP)

            # Updates sublogs
            sublogs.extend(new_sublogs) if new_sublogs else sublogs
            sublogs.remove(log)

            # Debug print to check the current state of sublogs
            logging.debug(f"Current sublogs: {sublogs}")

    def recursion_step(self, log: List[Tuple[str]]):
        """
        Single recursion step of Inductive Miner.

        Parameters:
            log: sublog as subset of the original event log
        """
        # Update the directly-follows graph (dfg), start_activities, and end_activities for the current sublog
        dfg, start_activities, end_activities = self._get_dfg(log)
        groups, new_sublogs = [], []
        operation_found = True

        # Check for base cases and build the corresponding part of the process tree
        base_cut, operator = self._handle_base_cases(log)
        if base_cut:  # If not a base case, apply different types of cuts
            self._build_process_tree(base_cut, operator)
        else:  # try to split log based on operator and build corresponding part of the process tree
            groups, operator = self._apply_cut(log, dfg, start_activities, end_activities)
            # Add the new sublogs to the list if not fall through case
            if operator != CutType.NONE:
                new_sublogs = self._split_log(log, groups, operator)
                logging.debug(f"Splitting log: {log} with groups: {groups} and operator: {operator}")
                logging.debug(f"New sublogs: {new_sublogs}")
                # Build the corresponding part of the process tree
                self._build_process_tree(groups, operator)
            else:  # If fall through case, build the flower model
                operation_found = False

        return operation_found, groups, new_sublogs

    def _get_dfg(self, log: List[Tuple]) -> Tuple[Dict[Tuple[str, str], int], Dict[str, int], Dict[str, int]]:
        """
        Builds the directly-follows graph (dfg) from the event log and extracts the start and end activities.

        Parameters:
            log: List of traces

        Returns:
            Tuple containing the dfg, start activities, and end activities.
        """
        dfg = defaultdict(int)
        start_activities = defaultdict(int)
        end_activities = defaultdict(int)

        for trace in log:
            start_activities[trace[0]] += 1
            end_activities[trace[-1]] += 1

            for i in range(len(trace) - 1):
                pair = (trace[i], trace[i + 1])
                dfg[pair] += 1

        return dfg, start_activities, end_activities

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

    def _is_nontrivial(self, max_groups: Optional[List[Set[str]]]) -> bool:
        """
        Checks if the number of groups is greater than 1 (i.e. the cut would separate the dfg further).

        Parameters:
            max_groups: List of groups of activities that form the cut.

        Returns:
            True if the cut is nontrivial, False otherwise.
        """
        return len(max_groups) > 1 if max_groups else False

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
        xor_cut = self._xor_cut(dfg, start_activities, end_activities)
        sequence_cut = self._sequence_cut(dfg, start_activities, end_activities)
        parallel_cut = self._parallel_cut(dfg, start_activities, end_activities)
        loop_cut = self._loop_cut(dfg, start_activities, end_activities)

        # Debug prints to check the cuts
        logging.debug(f"XOR Cut: {xor_cut}")
        logging.debug(f"Sequence Cut: {sequence_cut}")
        logging.debug(f"Parallel Cut: {parallel_cut}")
        logging.debug(f"Loop Cut: {loop_cut}")

        # If a nontrivial cut (>1) is found, return the partition and the corresponding operator
        if self._is_nontrivial(xor_cut):
            logging.debug("Applying XOR Cut")
            return xor_cut, CutType.XOR
        elif self._is_nontrivial(sequence_cut):
            logging.debug("Applying Sequence Cut")
            return sequence_cut, CutType.SEQUENCE
        elif self._is_nontrivial(parallel_cut):
            logging.debug("Applying Parallel Cut")
            return parallel_cut, CutType.PARALLEL
        elif self._is_nontrivial(loop_cut):
            logging.debug("Applying Loop Cut")
            return loop_cut, CutType.LOOP
        else:  # If no nontrivial cut is found, apply the fall-through case (flower model)
            logging.debug("Applying Fall-Through Case")
            flower_groups = self._handle_fall_through(log)
            return flower_groups, CutType.NONE

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
        search_str = ', '.join([', '.join(activity) for group in groups for activity in group if activity != self.TAU])
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
            match = self._find_subsequence_in_arbitrary_order(tree, search_str)
            tree = tree.replace(match, f'{new_cut_str}')

        # Update the process tree
        self.process_tree_str = tree

        return self.process_tree_str

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
        # Create a group per activity
        alphabet = set(a for (a, b) in dfg).union(set(b for (a, b) in dfg))
        transitive_predecessors, transitive_successors = self._get_transitive_relations(dfg)
        groups = [{a} for a in alphabet]
        if len(groups) == 0:
            return []

        # Merge pairwise reachable nodes (based on transitive relations)
        old_size = None
        while old_size != len(groups):
            old_size = len(groups)
            groups = self._merge_groups(groups, transitive_successors)

        # Merge pairwise unreachable nodes (based on transitive relations)
        old_size = None
        while old_size != len(groups):
            old_size = len(groups)
            groups = self._merge_groups(groups, transitive_predecessors)

        # Sort the groups based on their reachability
        groups = list(sorted(groups, key=lambda g: len(
            transitive_predecessors[next(iter(g))]) + (len(alphabet) - len(transitive_successors[next(iter(g))]))))

        return groups if len(groups) > 1 else []

    def _get_transitive_relations(self, dfg: Dict[Tuple[str, str], int]) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
        """
        Computes the transitive predecessors and successors for the directly-follows graph (dfg).

        Parameters:
            dfg: Directly-follows graph

        Returns:

        """
        transitive_predecessors = defaultdict(set)
        transitive_successors = defaultdict(set)

        for (a, b) in dfg:
            transitive_successors[a].add(b)
            transitive_predecessors[b].add(a)

        def dfs(node, graph):
            visited = set()
            stack = [node]
            while stack:
                current = stack.pop()
                for neighbor in graph[current]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        stack.append(neighbor)
            return visited

        nodes = list(transitive_successors.keys())
        for node in nodes:
            transitive_successors[node] = dfs(node, transitive_successors)

        nodes = list(transitive_predecessors.keys())
        for node in nodes:
            transitive_predecessors[node] = dfs(node, transitive_predecessors)

        return transitive_predecessors, transitive_successors

    def _merge_groups(self, groups, trans_succ):
        i = 0
        while i < len(groups):
            j = i + 1
            while j < len(groups):
                if self._check_merge_condition(groups[i], groups[j], trans_succ):
                    groups[i] = groups[i].union(groups[j])
                    del groups[j]
                    continue
                j = j + 1
            i = i + 1
        return groups

    def _check_merge_condition(self, g1, g2, trans_succ):
        for a1 in g1:
            for a2 in g2:
                if (a2 in trans_succ[a1] and a1 in trans_succ[a2]) or (
                        a2 not in trans_succ[a1] and a1 not in trans_succ[a2]):
                    return True
        return False

    def _xor_cut(self, dfg: Dict[Tuple[str, str], int], start: Dict[str, int], end: Dict[str, int]) -> List[Set[str]]:
        """
        Applies the XOR cut to the directly-follows graph (dfg).

        Parameters:
            dfg: Directly-follows graph
            start: Start activities in the log
            end: End activities in the log

        Returns:
            List of groups of activities that form the XOR cut.
        """
        # Convert DFG to undirected graph
        undirected_dfg = defaultdict(set)
        for (a, b) in dfg:
            undirected_dfg[a].add(b)
            undirected_dfg[b].add(a)

        for activity in start.keys():
            undirected_dfg[activity]

        for activity in end.keys():
            undirected_dfg[activity]

        # Detect connected components in the undirected graph
        def dfs(activity, component):
            stack = [activity]
            while stack:
                node = stack.pop()
                if node not in visited:
                    visited.add(node)
                    component.add(node)
                    for neighbor in undirected_dfg[node]:
                        if neighbor not in visited:
                            stack.append(neighbor)

        visited = set()
        components = []

        for activity in undirected_dfg:
            if activity not in visited:
                component = set()
                dfs(activity, component)
                components.append(component)

        return components if len(components) > 1 else []

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

        # Merge start and end activities into the first group (do-group)
        groups = [start_activities.union(end_activities)]

        # Extract inner edges (excluding start and end activities)
        inner_edges = [edge for edge in edges if edge[0] not in groups[0] and edge[1] not in groups[0]]

        # Create a group for the inner edges (loop group)
        groups.append(set([activity for edge in inner_edges for activity in edge]))

        # Exclude sets that are non-reachable from start/end activities from the loop groups
        def exclude_non_reachable(groups):
            group_a, group_b = set(), set()
            for group in groups:
                group_a = group if a in group else group_a
                group_b = group if b in group else group_b
            groups = [group for group in groups if group != group_a and group != group_b]
            groups.insert(0, group_a.union(group_b))

        pure_start_activities = start_activities.difference(end_activities)  # only start, not end activity
        # Put all activities in the do-group that follow a start activity which is not an end activity
        # (a loop activity can only follow a start activity if it is also an end activity)
        for a in pure_start_activities:
            for (x, b) in edges:
                if x == a:
                    exclude_non_reachable(groups)

        pure_end_activities = end_activities.difference(start_activities)  # only end, not start activity
        # Put all activities in the do-group that precede an end activity which is not a start activity
        # (a loop activity can only precede an end activity if it is also a start activity)
        for b in pure_end_activities:
            for (a, x) in edges:
                if x == b:
                    exclude_non_reachable(groups)
        # Check start completeness
        # All loop activities must be able to reach the start activities
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
        # All loop activities must be able to be reached from the end activities
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

        # Return the cut if more than one group (i.e., do- and loop-group found)
        groups = [group for group in groups if group != set()]
        return groups if len(groups) > 1 else []

    def _split_log(self, log: List[Tuple[str]], cut: List[Set[str]], operator: CutType) -> List[List[Tuple[str, ...]]]:
        if operator == CutType.SEQUENCE:
            return self._projection_split(log, cut)
        elif operator == CutType.XOR:
            return self._xor_split(log, cut)
        elif operator == CutType.PARALLEL:
            return self._projection_split(log, cut)
        elif operator == CutType.LOOP:
            return self._loop_split(log, cut)
        else:
            return []

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

    def _projection_split(self, log: List[Tuple[str]], cut: List[Set[str]]) -> List[List[Tuple[str, ...]]]:
        """
        Splits the event log based on the groups provided.

        Parameters:
            log: List of traces
            cut: List of groups of activities that form the cut

        Returns:
            List of sublogs resulting from the parallel split.
        """
        # Initialize a dictionary to store the sublogs
        sublogs_dict = {}

        # Iterate over each trace in the log
        for trace in log:
            # Iterate over each group in the cut
            for i, group in enumerate(cut):
                subtrace = []

                # Iterate over each activity in the trace
                for activity in trace:
                    if activity in group:
                        subtrace.append(activity)

                # If no activities were added to the subtrace, add an empty trace
                if not subtrace:
                    subtrace.append('')

                # Add the subtrace to the corresponding group in the sublogs dictionary
                sublogs_dict[str(group)] = sublogs_dict.get(str(group), []) + [tuple(subtrace)]

        # Convert the sublogs dictionary to a list of sublogs
        sublogs = [log for log in sublogs_dict.values()]

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
            sublog = []
            # Iterate over the old traces (traces in the log)
            for i, old_trace in enumerate(old_traces):
                # Find subsequences in the old trace that match the new trace and add them to the sublog
                found_subtrace = True
                while found_subtrace:
                    found_subtrace = False
                    # try with subsequences of the new trace of decreasing length
                    for j in range(0, len(new_trace)):
                        subtrace = self._find_subsequence_in_arbitrary_order(old_trace, new_trace[j:])
                        if len(subtrace) > 0:  # if a subsequence is found
                            found_subtrace = True
                            sublog.append(tuple(subtrace))
                            # Remove the found subsequence from the old trace
                            # TODO: breaks if activity name includes '?'
                            old_trace = old_trace.replace(subtrace, '?', 1)
                            old_traces[i] = old_trace
                            break
                        j += 1

            # Add the sublog to the new log
            if sublog:
                new_log.append(sublog)

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
        Visualizes the process tree as a PNG image.

        This method parses the process tree string generated by the Inductive Miner algorithm,
        constructs a graph using Graphviz, and displays the resulting image.
        """
        def parse_tree(tree_str):
            """
            Parses the process tree string into a nested structure.

            Parameters:
                tree_str (str): The process tree string.

            Returns:
                tuple: A nested structure representing the process tree.
            """
            stack = []
            node_counter = 0

            for char in tree_str:
                if char == '(':
                    stack.append(char)
                elif char == ')':
                    children = []
                    while stack and stack[-1] != '(':
                        children.append(stack.pop())
                    stack.pop()  # remove '('
                    operator = stack.pop()
                    node_id = f'node{node_counter}'
                    node_counter += 1
                    stack.append((operator, children[::-1], node_id))
                elif char not in ' ,':
                    stack.append(char)

            return stack[0] if stack else None

        def create_graph(node, graph):
            """
            Recursively creates nodes and edges in the Graphviz graph.

            Parameters:
                node (tuple): The current node in the parsed tree.
                graph (graphviz.Digraph): The Graphviz graph being constructed.
            """
            if isinstance(node, tuple):
                operator, children, node_id = node
                graph.node(node_id, operator, shape='circle')
                for child in children:
                    if isinstance(child, tuple):
                        child_id = child[2]
                        create_graph(child, graph)
                    else:
                        child_id = f'node{child}'
                        graph.node(child_id, child, shape='box')
                    graph.edge(node_id, child_id)
            else:
                node_id = f'node{node}'
                graph.node(node_id, node, shape='box')

        # Get the process tree string from the Inductive Miner instance
        tree_str = self.process_tree_str

        # Parse the process tree string into a nested structure
        tree = parse_tree(tree_str)

        # Create a new Graphviz graph
        graph = graphviz.Digraph(format='png')

        # Populate the graph with nodes and edges
        create_graph(tree, graph)

        # Render the graph to a PNG file in memory and display it
        graph_bytes = graph.pipe(format='png')
        display(Image(graph_bytes))

    def build_and_visualize_petrinet(self):
        """
        Builds and visualizes the Petri net from the event log using PM4Py.

        Uses the inductive miner algorithm to discover the Petri net.
        """
        if self.net is None or self.initial_marking is None or self.final_marking is None:
            # Convert the event log to PM4Py event log format
            data = [{'case:concept:name': idx, 'concept:name': activity, 'time:timestamp': idx}
                    for idx, trace in enumerate(self.event_log)
                    for activity in trace]
            df = pd.DataFrame(data)
            event_log = log_converter.apply(df)

            # Discover ProcessTree using inductive miner algorithm
            process_tree = inductive_miner.apply(event_log)

            # Convert ProcessTree to PetriNet
            self.net, self.initial_marking, self.final_marking = pt_to_petri_converter.apply(process_tree)

        # Visualize the Petri net
        gviz = pn_vis.apply(self.net, self.initial_marking, self.final_marking)
        pn_vis.view(gviz)

    """
    UNUSED CODE @Dilemmaqwer
    
    
    def _invert_graph(graph: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
        inverted = defaultdict(set)
        all_nodes = set(graph.keys())
        for src in graph:
            for dest in graph[src]:
                inverted[dest].add(src)
            all_nodes.update(graph[src])

        for node in all_nodes:
            if node not in inverted:
                inverted[node] = set()

        return inverted
    
    def _sequence_cut(dfg: Dict[Tuple[str, str], int], start: Dict[str, int], end: Dict[str, int]) -> List[Set[str]]:
        adj_list = defaultdict(set)
        for (a, b) in dfg:
            adj_list[a].add(b)

        sccs = _strongly_connected_components(adj_list)

        scc_map = {}
        for scc in sccs:
            for node in scc:
                scc_map[node] = scc

        scc_graph = defaultdict(set)
        for (a, b) in dfg:
            if scc_map[a] != scc_map[b]:
                scc_graph[frozenset(scc_map[a])].add(frozenset(scc_map[b]))

        in_degree = defaultdict(int)
        for src in scc_graph:
            for dest in scc_graph[src]:
                in_degree[dest] += 1

        zero_in_degree = [node for node in scc_graph if in_degree[node] == 0]
        topo_sorted_sccs = []
        while zero_in_degree:
            node = zero_in_degree.pop()
            topo_sorted_sccs.append(node)
            for neighbor in scc_graph[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    zero_in_degree.append(neighbor)

        first_part = set()
        second_part = set()
        in_first_part = True
        for scc in topo_sorted_sccs:
            if in_first_part:
                first_part.update(scc)
                if any(node in end for node in scc):
                    in_first_part = False
            else:
                second_part.update(scc)

        return [first_part, second_part]

    def _strongly_connected_components(self, graph: Dict[str, Set[str]]) -> List[Set[str]]:
        index = 0
        stack = []
        indices = {}
        lowlinks = {}
        on_stack = defaultdict(bool)
        sccs = []

        def strongconnect(node):
            nonlocal index
            indices[node] = index
            lowlinks[node] = index
            index += 1
            stack.append(node)
            on_stack[node] = True

            for neighbor in graph[node]:
                if neighbor not in indices:
                    strongconnect(neighbor)
                    lowlinks[node] = min(lowlinks[node], lowlinks[neighbor])
                elif on_stack[neighbor]:
                    lowlinks[node] = min(lowlinks[node], indices[neighbor])

            if lowlinks[node] == indices[node]:
                scc = set()
                while True:
                    w = stack.pop()
                    on_stack[w] = False
                    scc.add(w)
                    if w == node:
                        break
                sccs.append(scc)

        for node in graph:
            if node not in indices:
                strongconnect(node)

        return sccs"""

if __name__ == '__main__':
     #real log
     #format log
     log = utils.import_csv(FILE_PATH_CSV)
     log = pm4py.format_dataframe(log, case_id='case_id', activity_key='activity', timestamp_key='timestamp')

     # IM
     process_tree_IM = inductive_miner.apply(log, variant=inductive_miner.Variants.IM)
     # print('process_tree_IM ===== ', process_tree_IM)
     gviz_tree_IM = pt_vis.apply(process_tree_IM)
     pt_vis.view(gviz_tree_IM)

     net, initial_marking, final_marking = pt_to_petri_converter.apply(process_tree_IM)
     gviz_petri_IM = pn_vis.apply(net, initial_marking, final_marking)
     pn_vis.view(gviz_petri_IM)

     # IMf
     process_tree_IMf = inductive_miner.apply(log, parameters=parameters,variant=inductive_miner.Variants.IMf)
     gviz_tree_IMf = pt_visualizer.apply(process_tree_IMf)
     pt_visualizer.view(gviz_tree_IMf)

     net, initial_marking, final_marking = pt_to_petri_converter.apply(process_tree_IMf)
     gviz_petri_IMf = pn_vis.apply(net, initial_marking, final_marking)
     pn_vis.view(gviz_petri_IMf)