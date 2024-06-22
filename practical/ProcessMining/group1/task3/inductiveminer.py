import logging
import os
import re
from collections import defaultdict
from enum import Enum
from typing import List, Tuple, Dict, Set, Optional, Union
import graphviz
import networkx as nx
import pandas as pd
from IPython.display import Image, display
import pm4py
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.visualization.petri_net import visualizer as pn_vis
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.objects.conversion.process_tree import converter as pt_to_petri_converter
from practical.ProcessMining.group1.shared.visualizer import Visualizer

logging.basicConfig(level="INFO")  # Change to DEBUG for prints


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

    def __init__(self, event_log: Optional[Union[List[Tuple[str]], str]] = None):
        """
        Initialize the Inductive Miner with an event log.

        Parameters:
            event_log: List of traces
        """
        if isinstance(event_log, str):
            self.event_log = self._import_event_log(event_log)
        else:
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

    def _import_event_log(self, file_path: str, case_id='case_id', activity_key='activity',
                          timestamp_key='timestamp') -> List[Tuple[str]]:
        """
        Imports the event log from a file.

        Parameters:
            file_path (str): The path to the event log file.

        Returns:
            Tuple[pd.DataFrame, Dict[int, str], List[Tuple[int, int]]]: The event log data,
            the mapping of activity IDs to activity names, and all pairs of activities.
        """
        # Check if the file exists
        if not os.path.exists(file_path):
            raise Exception("File does not exist")

        # Differentiate between CSV and XES files
        extension = os.path.splitext(file_path)[1]
        if extension == '.csv':
            event_log = pd.read_csv(file_path, sep=';')
            event_log = pm4py.format_dataframe(event_log, case_id=case_id, activity_key=activity_key,
                                               timestamp_key=timestamp_key)
        elif extension == '.xes':
            event_log = pm4py.read_xes(file_path)
        else:
            raise Exception("File extension must be .csv or xes")

        # Sort the event log by case ID and timestamp
        event_log = event_log.sort_values(['case:concept:name', 'time:timestamp'])
        # Group the event log by case ID and extract the activities as tuples
        event_log = event_log.groupby('case:concept:name')['concept:name'].apply(tuple).reset_index()
        event_log = event_log['concept:name'].tolist()
        return event_log

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
        base_activity = {alphabet[0]} if alphabet else set()
        len_base_activity = len(alphabet[0])
        tau_activity = set(self.TAU)
        operator = CutType.NONE
        groups = []

        # If there is only one unique activity in the log
        if len(alphabet) == 1:
            if all(len(trace) == len_base_activity for trace in traces):  # exactly once (1)
                operator = CutType.NONE
                groups = [base_activity]
            elif all(len(trace) == 0 or len(trace) == len_base_activity for trace in traces):  # never or once (0,1)
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
        flower_groups = [set(self.TAU)] + [{activity} for activity in sorted(list(self._get_alphabet(log)))]
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

        groups = [sorted([activity for activity in group]) for group in groups]
        # Sort the groups for cut types where order of groups is not important to make the output deterministic
        if cut_type == CutType.XOR or cut_type == CutType.PARALLEL:
            groups = sorted(groups, key=lambda x: x[0])
        if cut_type == CutType.LOOP:
            groups = sorted(groups[:1]) + sorted(groups[1:], key=lambda x: x[0])

        # Convert the groups into a string
        group_str = ', '.join([', '.join(group) for group in groups])
        search_str = ', '.join([', '.join(group) for group in groups if group != [self.TAU]])
        # Get the string representation of the cut type
        cut_str = f'{cut_type.value}' if cut_type != CutType.NONE else ''

        # If the group has more than one activity, wrap it in parentheses and prepend the cut type
        if len(groups) > 1 or (len(groups) == 1 and len(groups[0]) > 1):
            new_cut_str = f'{cut_str}({group_str})'
        else:  # If the group has only one activity, prepend the cut type
            new_cut_str = f'{cut_str}{group_str}'

        # If the current process tree is empty (no cut applied yet), replace it with the new cut string
        if tree == '()':
            tree = new_cut_str
        else:
            # If the current process tree is not empty, find the subsequence in the tree that matches the group string
            # and replace it with the new cut string
            match = self._find_substring_in_arbitrary_order(tree, search_str)
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
        alphabet = self._get_alphabet_from_dfg(dfg)
        groups = [{activity} for activity in alphabet]

        # Initialize directed graph
        directed_graph = nx.DiGraph()
        directed_graph.add_edges_from(dfg)

        # Compute transitive predecessors and successors
        transitive_successors = {node: set(nx.descendants(directed_graph, node)) for node in directed_graph.nodes}
        transitive_predecessors = {node: set(nx.ancestors(directed_graph, node)) for node in directed_graph.nodes}

        # Function to merge groups based on transitive relations
        def merge_groups(groups, transitive_relations):
            merged = True
            while merged:
                merged = False
                for i in range(len(groups)):
                    for j in range(i + 1, len(groups)):
                        if has_transitive_relation(groups[i], groups[j], transitive_relations):
                            groups[i] = groups[i].union(groups[j])
                            groups.pop(j)
                            merged = True
                            break
                    if merged:
                        break
            return groups

        # Function to check if two groups should be merged
        def has_transitive_relation(group_a, group_b, transitive_relations):
            return any(
                (b in transitive_relations[a] and a in transitive_relations[b]) or
                (b not in transitive_relations[a] and a not in transitive_relations[b])
                for a in group_a for b in group_b
            )

        # Merge pairwise reachable nodes (based on transitive relations)
        groups = merge_groups(groups, transitive_successors)
        # Merge pairwise unreachable nodes (based on transitive relations)
        groups = merge_groups(groups, transitive_predecessors)

        # Sort the groups based on their reachability
        groups.sort(key=lambda g: len(transitive_predecessors[next(iter(g))]) + (
                len(alphabet) - len(transitive_successors[next(iter(g))])))

        return groups

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
        # Convert DFG to an undirected networkx graph
        undirected_graph = nx.Graph()
        undirected_graph.add_nodes_from(self._get_alphabet_from_dfg(dfg).union(start.keys()).union(end.keys()))
        undirected_graph.add_edges_from(dfg.keys())

        # Detect connected components in the graph
        components = list(nx.connected_components(undirected_graph))
        # Replace empty component with the TAU activity
        groups = [component if component != {''} else set(self.TAU) for component in components]

        return groups

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
        edges = set(dfg.keys())
        start_activities = set(start.keys())
        end_activities = set(end.keys())

        # Initialize groups with individual activities
        all_activities = {activity for edge in edges for activity in edge}
        groups = [{activity} for activity in all_activities]

        # Determine if two groups are not connected by edges
        def should_merge(group_a, group_b):
            return any((a, b) not in edges or (b, a) not in edges for a in group_a for b in group_b)

        merged = True
        while merged:
            merged = False
            for i in range(len(groups)):
                for j in range(i + 1, len(groups)):
                    if should_merge(groups[i], groups[j]):
                        groups[i] = groups[i].union(groups[j])
                        groups.pop(j)
                        merged = True
                        break
                if merged:
                    break

        # Filter out groups that do not contain start and end activities
        groups = [group for group in groups if group & start_activities and group & end_activities]

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
        # Extract edges and activities
        edges = list(dfg.keys())
        start_activities = set(start.keys())
        end_activities = set(end.keys())

        # Merge start and end activities into the first group (do-group)
        do_group = start_activities.union(end_activities)

        # Extract inner edges (excluding start and end activities)
        inner_edges = [edge for edge in edges if edge[0] not in do_group and edge[1] not in do_group]

        # Add connected components as loop groups
        graph = nx.Graph()
        graph.add_edges_from(inner_edges)
        connected_components = list(nx.connected_components(graph))
        loop_groups = connected_components if len(connected_components) > 0 else []

        # Add remaining activities to their own groups
        alphabet = self._get_alphabet_from_dfg(dfg)
        loop_groups += [{a} for a in alphabet if not any(a in group for group in loop_groups + [do_group])]

        pure_start_activities = start_activities.difference(end_activities)  # only start, not end activity
        # Put all activities in the do-group that follow a start activity which is not an end activity
        # (a loop activity can only follow a start activity if it is also an end activity)
        for a in pure_start_activities:
            for (x, b) in edges:
                if x == a:
                    # Find the group that contains the activity
                    merge_group = set()
                    for group in loop_groups:
                        merge_group = group if b in group else merge_group
                    # Remove the group from the loop groups and add it to the do group
                    loop_groups = [group for group in loop_groups if group != merge_group]
                    do_group = do_group.union(merge_group)

        pure_end_activities = end_activities.difference(start_activities)  # only end, not start activity
        # Put all activities in the do-group that precede an end activity which is not a start activity
        # (a loop activity can only precede an end activity if it is also a start activity)
        for b in pure_end_activities:
            for (a, x) in edges:
                if x == b:
                    # Find the group that contains the activity
                    merge_group = set()
                    for group in loop_groups:
                        merge_group = group if b in group else merge_group
                    # Remove the group from the loop groups and add it to the do group
                    loop_groups = [group for group in loop_groups if group != merge_group]
                    do_group = do_group.union(merge_group)

        # Ensure all loop activities can reach start activities
        i = 1
        while i < len(loop_groups):
            if any((a, b) not in edges for a in loop_groups[i] for b in start_activities):
                do_group.update(loop_groups.pop(i))
            else:
                i += 1

        # Ensure all loop activities can be reached from end activities
        i = 1
        while i < len(loop_groups):
            if any((b, a) not in edges for a in loop_groups[i] for b in end_activities):
                do_group.update(loop_groups.pop(i))
            else:
                i += 1

        # Return the cut if more than one group (i.e., do- and loop-group found)
        groups = [do_group, *loop_groups]
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
                # Iterate over each activity in the trace
                subtrace = [activity for activity in trace if activity in group]

                # If no activities were added to the subtrace, add an empty trace
                if not subtrace:
                    subtrace = ['']

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
        new_log = []

        # Iterate over each group in the cut
        for group in cut:
            sublog = []
            # Iterate over each trace in the log
            for trace in log:
                trace_list = list(trace)  # Convert the trace tuple to a list for manipulation
                # Continuously find and remove subtraces that match the current group
                while True:
                    subtrace = self._find_subsequence_in_arbitrary_order(trace_list, list(group))
                    if not subtrace:
                        break
                    sublog.append(tuple(subtrace))  # Add the found subsequence to the sublog
                    # Remove the elements of the found subsequence from the trace_list
                    for item in subtrace:
                        trace_list.remove(item)
            if sublog:
                new_log.append(sublog)

        return new_log

    def _find_subsequence_in_arbitrary_order(self, main: List[str], sub: List[str]) -> List[str]:
        """
        Finds a subsequence in the main list that contains the same elements as the sub list,
        regardless of their order.

        Parameters:
            main: The main list in which to find the subsequence.
            sub: The sub list whose elements should be found in the main list.

        Returns:
            The found subsequence in the main list. If no such subsequence is found, returns an empty list.
        """
        sub_len = len(sub)
        sorted_sub = sorted(sub)  # Sort the sub list for comparison

        # Slide a window over the main list to find a matching subsequence
        for i in range(len(main) - sub_len + 1):
            window = main[i:i + sub_len]
            if sorted(window) == sorted_sub:  # Check if the sorted window matches the sorted sub list
                return window
        return []

    def _find_substring_in_arbitrary_order(self, main: str, sub: str) -> str:
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
        constructs a graph using Graphviz, and returns the resulting image.

        Returns:
            Image: The PNG image of the process tree.
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

            # Use regex to find tokens: sequences of characters, parentheses, and commas
            tokens = re.findall(r'→|↺|∧|×|\(|\)|,|[^,()\s][^,()]*', tree_str)

            for token in tokens:
                if token == '(':
                    stack.append(token)  # Add opening bracket to the stack
                elif token == ')':
                    children = []
                    while stack and stack[-1] != '(':
                        children.append(stack.pop())  # Collect all children until an opening bracket is found
                    stack.pop()  # Remove the opening bracket
                    operator = stack.pop()  # Get the operator
                    node_id = f'node{node_counter}'  # Generate a unique node ID
                    node_counter += 1
                    stack.append((operator.strip(), children[::-1], node_id))  # Append the parsed subtree
                elif token == ',':
                    continue  # Skip commas
                else:
                    stack.append(token)  # Add activity or operator to the stack

            return stack[0] if stack else None  # Return the root of the parsed tree

        # Parse the process tree string into a nested structure
        tree = parse_tree(self.process_tree_str)
        # print(f"Parsed tree: {tree}")  # show the parsed tree if needed

        # Create a new Graphviz graph
        visualizer = Visualizer()
        graph = visualizer.get_process_tree(tree)

        return graph

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
            event_log = log_converter.apply(df)  # Convert the DataFrame to an event log

            # Discover ProcessTree using inductive miner algorithm
            process_tree = inductive_miner.apply(event_log)

            # Convert ProcessTree to PetriNet
            self.net, self.initial_marking, self.final_marking = pt_to_petri_converter.apply(process_tree)

        # Visualize the Petri net
        visualizer = Visualizer()
        graph = visualizer.get_petri_net(self.net, self.initial_marking, self.final_marking)
        return graph

    def get_petrinet(self) -> Tuple[nx.DiGraph, Dict[str, int], Dict[str, int]]:
        """
        Returns the Petri net and initial/final markings.
        """
        return self.net, self.initial_marking, self.final_marking
