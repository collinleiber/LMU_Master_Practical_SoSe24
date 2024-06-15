import os
import copy
import itertools
from practical.ProcessMining.group2.inductive_miner.src.graph_utils import *

script_dir = os.path.dirname(os.path.abspath(__file__))


class EventLog:
    # Priority of given traces over file path
    def __init__(self, traces):
        self.traces = traces

    @classmethod
    def from_file(cls, file_path=None): 
        with open(os.path.join(script_dir, file_path), 'r') as file:
            traces = dict()
            for line in file:
                trace = line.strip()
                if trace in traces:
                    traces[trace] += 1
                else:
                    traces[trace] = 1
            return EventLog(traces)

class DirectlyFollowsGraph(Graph):
    def __init__(self, event_log: EventLog):
        self.event_log = event_log
        self.graph = defaultdict(list)  # Adjacency list, format: "node_id" : [children]
        self.start_nodes = set()
        self.end_nodes = set()

    def construct_dfg(self):
        for trace in self.event_log.traces.keys():
            if trace:  # Check if trace is not empty
                self.start_nodes.add(trace[0])  # First activity in trace is a start
                self.end_nodes.add(trace[-1])  # Last activity in trace is an end

                for i in range(len(trace) - 1):
                    current_activity = trace[i]
                    next_activity = trace[i + 1]

                    if next_activity not in self.graph[current_activity]:
                        self.graph[current_activity].append(next_activity)
                    if (next_activity not in self.graph):  # Add next activity to graph if not already present, i.e. if it is an end node
                        self.graph[next_activity] = []
                
                for node in self.start_nodes:
                    if node not in self.graph: # Isolated nodes
                        self.graph[node] = []
            else:
                self.graph[''] = [] # Add tau node for empty traces

    def __str__(self):
        return f"Directly Follows Graph: (\n\tGraph: {dict(self.graph)}\n\tStart nodes: {self.start_nodes}\n\tEnd nodes: {self.end_nodes}\n)"

class ProcessTree:
    def __init__(self, event_log: EventLog):
        self.event_log = event_log
        self.root = None
        self.children = []

    def find_base_case(self):
        if len(self.event_log.traces) == 0:
            return ()
        elif len(self.event_log.traces) == 1:
            only_trace = next(iter(self.event_log.traces))
            if only_trace == "":
                return 'tau'
            elif len(only_trace) == 1:
                return only_trace
        return None

    def find_exclusive_choice_cut(self, dfg: DirectlyFollowsGraph):
        # Convert the graph to undirected
        undirected = dfg.convert_to_undirected()
        # Find connected components
        cuts = undirected.find_components()

        return None if len(cuts) == 1 else cuts
    
    def find_sequence_cut(self, dfg: DirectlyFollowsGraph):
        def is_skippable(p: int, cuts: list) -> bool:
            # Check for skippable activities (according to strict sequence cut detection)
            # This can be helpful if optionality in sequence is present
            edges = dfg.get_all_edges()
            for i, j in itertools.product(range(p), range(p + 1, len(sorted_cuts))):
                for node1, node2 in itertools.product(sorted_cuts[i], sorted_cuts[j]):
                    transitive_edge = (node1, node2)
                    if (transitive_edge in edges 
                        or node1 in dfg.end_nodes 
                        or node2 in dfg.start_nodes):
                        return True
            return False

        remaining_nodes = set(dfg.get_all_nodes())
        cuts = []

        # Find strongly connected components
        components = dfg.find_strongly_con_components()
        for component in components:
            if len(component) > 1:
                cuts.append(component)
                for node in component:
                    remaining_nodes.discard(node)

        # Find pairwise unreachable nodes and merge them to one node
        unreachable_pairs = dfg.find_unreachable_pairs()
        merged_nodes = list({node for pair in unreachable_pairs for node in pair})
        if merged_nodes:
            cuts.append(merged_nodes)

        # Remove merged nodes from remaining nodes to be processed
        for node in merged_nodes:
            remaining_nodes.discard(node)

        for node in remaining_nodes:
            cuts.append([node])

        # Build cuts graph
        cuts_graph, cut_map = dfg.build_cuts_graph(cuts)

        # Sort cuts by traversing the cut graph (always a path graph)
        start_node = cut_map[list(dfg.start_nodes)[0]]
        sorted_cut_indices = cuts_graph.traverse_path(start_node)
        sorted_cuts = [cuts[i] for i in sorted_cut_indices]

        # Merge skippable cuts
        merged_cuts = []
        i = 0
        while i < len(sorted_cuts):
            # If the current cut is skippable, start merging process
            if is_skippable(i, sorted_cuts):
                start = i
                while i < len(sorted_cuts) and is_skippable(i, sorted_cuts):
                    i += 1
                # Merge all consecutive skippable cuts into one
                merged_cuts.append([activity for cut in sorted_cuts[start:i] for activity in cut])
            else:
                # If the current cut is not skippable, just add it to the result
                merged_cuts.append(sorted_cuts[i])
                i += 1

        return None if len(sorted_cuts) == 1 else merged_cuts

    def find_parallel_cut(self, dfg: DirectlyFollowsGraph):
        # Mark edges to be removed
        edges = set(dfg.get_all_edges())
        removed_edges = set()
        for edge in edges:
            if edge[::-1] in edges:
                removed_edges.add(edge)

        # Insert edge where no edge exists between two nodes
        updated_graph = Graph(copy.deepcopy(dfg.graph))
        nodes = dfg.get_all_nodes()
        for node1 in nodes:
            for node2 in nodes:
                if (node1 != node2 
                        and node2 not in updated_graph.graph[node1] 
                        and (node1, node2) not in removed_edges
                    ):
                    updated_graph.graph[node1].append(node2)

        # Remove dual edges to be removed
        for edge in removed_edges:
            updated_graph.graph[edge[0]].remove(edge[1])

        cuts = updated_graph.find_components()

        # Check if each component contains both a start and end node
        for cut in cuts:
            if not dfg.start_nodes.intersection(set(cut)) or not dfg.end_nodes.intersection(set(cut)):
                return None

        return None if len(cuts) == 1 else cuts

    def find_loop_cut(self, dfg: DirectlyFollowsGraph):
        cuts = []
        # Create do-body, start with all start/end nodes
        start_nodes = dfg.start_nodes
        end_nodes = dfg.end_nodes
        start_end_nodes = start_nodes.union(end_nodes)
        do_body = start_end_nodes

        # Temporarily remove start/end activities
        reduced_graph = Graph(copy.deepcopy(dfg.graph))
        for node in start_end_nodes:
            if node in reduced_graph.graph.keys():
                del reduced_graph.graph[node]
            for children in reduced_graph.graph.values():
                if node in children:
                    children.remove(node)

        # Find connected components as possible candidates for loop bodies
        undirected = reduced_graph.convert_to_undirected()
        components = set(tuple(c) for c in undirected.find_components())

        # Find components that are connected to start nodes in do-body
        components_connected_to_start_nodes = set()
        connected_nodes_start = set()
        for component in components:
            for node in component:
                for neighbor in dfg.get_neighbors(node):
                    if neighbor in start_nodes:
                        components_connected_to_start_nodes.add(tuple(component))
                        connected_nodes_start.add(node)

        # Find components that are connected from end nodes in do-body
        components_connected_from_end_nodes = set()
        connected_nodes_end = set()
        for end_node in end_nodes:
            for component in components:
                for node in component:
                    if node in dfg.get_neighbors(end_node):
                        components_connected_from_end_nodes.add(tuple(component))
                        connected_nodes_end.add(node)

        # Find (invalid) components that are connected from non-end nodes in do-body
        for node in do_body - end_nodes:
            components_to_remove = set()
            
            for component in components:
                for neighbor in dfg.get_neighbors(node):
                    if neighbor in component and neighbor not in do_body:
                        # Merge with do-body
                        do_body.update(component)
                        components_to_remove.add(component)
                        break 

            components -= components_to_remove

        # Find (invalid) components that are connected to non-start nodes in do-body
        components_to_remove = set()

        for component in components:
            for node in component:
                for neighbor in dfg.get_neighbors(node):
                    if neighbor in do_body and neighbor not in start_nodes:
                        # Merge with do-body
                        do_body.update(component)
                        components_to_remove.add(component)
                        break

        components -= components_to_remove

        # Check connectivity completeness to start nodes
        for node in connected_nodes_start:
            for start_node in start_nodes:
                if start_node not in dfg.get_neighbors(node):
                    components_connected_to_start_nodes = {c for c in components_connected_to_start_nodes 
                                                           if node not in c}
                    break

        # Check reachability completeness from end nodes
        for node in connected_nodes_end:
            for end_node in end_nodes:
                if not node in dfg.get_neighbors(end_node):
                    components_connected_from_end_nodes = {c for c in components_connected_from_end_nodes 
                                                           if node not in c}
                    break

        # Remove components that are not connected to start nodes in do-body
        invalid_components = components - (components_connected_to_start_nodes.union(
            components_connected_from_end_nodes)
            )
        loop_bodies = components - invalid_components

        # Merge invalid components to do-body
        for component in (invalid_components):
            do_body.update(component)

        # Add valid do-bodies to cuts
        cuts.append(list(do_body)) # Do-body has to be the first element
        for loop_body in loop_bodies:
            cuts.append(list(loop_body))

        # We need at least to components for a valid loop cut
        return None if len(cuts) < 2 else cuts

    def exclusive_choice_split(self, cuts):
        cuts = [set(cut) for cut in cuts]
        splits = [set() for _ in range(len(cuts))]

        for trace in self.event_log.traces:
            if trace:
                for i, cut in enumerate(cuts):
                    if all(activity in cut for activity in trace):
                        splits[i].add(trace)
                        break

        splits = [list(split) for split in splits]

        return splits

    def sequence_split(self, cuts):
        cuts = [set(cut) for cut in cuts] # Convert to set for faster lookup
        splits = [set() for _ in range(len(cuts))]

        for trace in self.event_log.traces:
            subtrace = ""
            trace = list(trace)
            for i,cut in enumerate(cuts):
                while trace and trace[0] in cut:
                    subtrace += trace.pop(0)
                splits[i].add(subtrace)
                subtrace = ""
                if not trace:
                    break

        splits = [list(split) for split in splits]

        return splits

    def parallel_split(self, cuts):
        cuts = [set(cut) for cut in cuts]
        splits = [set() for _ in range(len(cuts))]

        for trace in self.event_log.traces:
            for cut in cuts:
                sub_trace = ''.join([activity for activity in trace if activity in cut])
                if sub_trace:
                    splits[cuts.index(cut)].add(sub_trace)

        splits = [list(split) for split in splits]

        return splits

    def loop_split(self, cuts):
        cuts = [set(cut) for cut in cuts]
        splits = [set() for _ in range(len(cuts))]

        for trace in event_log.traces:
            current_sub_trace = ""
            current_cut_index = -1
            
            for activity in trace:
                for cut_index, cut in enumerate(cuts):
                    if activity in cut:
                        if current_cut_index != cut_index:
                            if current_sub_trace:
                                splits[current_cut_index].add(current_sub_trace)
                                current_sub_trace = ""
                            current_cut_index = cut_index
                        current_sub_trace += activity
                        break
            
            if current_sub_trace:
                splits[current_cut_index].add(current_sub_trace)

        splits = [list(split) for split in splits]

        return splits
    
    def construct_process_tree(self):
        base_case = self.find_base_case()
        if base_case is not None:
            return base_case
        
        dfg = DirectlyFollowsGraph(self.event_log)
        dfg.construct_dfg()

        cut_methods = [
            (self.find_exclusive_choice_cut, self.exclusive_choice_split, 'X'),
            (self.find_sequence_cut, self.sequence_split, '->'),
            (self.find_parallel_cut, self.parallel_split, '||'),
            (self.find_loop_cut, self.loop_split, 'O')
        ]

        for find_cut, process_split, operator in cut_methods:
            cuts = find_cut(dfg)
            if cuts is not None:
                splits = process_split(cuts)
                subtrees = []
                for split in splits:
                    sublog = dict()
                    for trace in split:
                        sublog[trace] = 1
                    subtree = ProcessTree(EventLog(traces=sublog)).construct_process_tree()
                    subtrees.append(subtree)
                self.root = operator
                self.children = [subtree[0] for subtree in subtrees]
                return operator, subtrees
        
        # Fallthrough case
        return 'O', ['tau'] + dfg.get_all_nodes()
    
    def __str__(self) -> str:
        operator_map = {
            'O': '↺',
            'X': 'x',
            '->': '➜',
            '||': '∧',
            'tau': '𝝉'
        }
        tree = self.construct_process_tree()
        def print_tree(node) -> str:
            if isinstance(node, str):
                return node
            else:
                operator, children = node
                return f"{operator_map[operator]}({', '.join(print_tree(child) for child in children)})"
            
        return print_tree(tree)
            
class InductiveMiner():
    def __init__(self):
        pass

    def mine_process_model(self, event_log):
        # Construct Directly-Follows Graph (DFG)
        dfg = DirectlyFollowsGraph(event_log)
        dfg.construct_dfg()

        # Construct Process Tree from DFG
        process_tree = ProcessTree(event_log)

        return process_tree