import os
import copy
from practical.ProcessMining.group2.inductive_miner.src.graph_utils import *

script_dir = os.path.dirname(os.path.abspath(__file__))


class EventLog:
    # Priority of given traces over file path
    def __init__(self, traces=None, file_path=None):
        if file_path is not None:
            self.file_path = os.path.join(script_dir, file_path)
            self.traces = {}
            self.load_from_file()
        elif traces is not None:
            self.traces = traces
        else:
            raise ValueError("No file_path or traces given for EventLog.")

    @classmethod
    def from_file(cls, file_path):
        return cls(file_path=file_path)

    @classmethod
    def from_traces(cls, traces):
        return cls(traces=traces)

    def load_from_file(self):  # TODO - what is the format of the log file?
        with open(self.file_path, 'r') as file:
            for line in file:
                trace = line.strip()
                if trace in self.traces:
                    self.traces[trace] += 1
                else:
                    self.traces[trace] = 1

    # TODO - other input methods?


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
                    if (
                        next_activity not in self.graph
                    ):  # Add next activity to graph if not already present, i.e. if it is an end node
                        self.graph[next_activity] = []

    # Debugging helper
    def print_graph(self):
        print("Graph: ", dict(self.graph))
        print("Start nodes: ", self.start_nodes)
        print("End nodes: ", self.end_nodes)


class ProcessTree:
    def __init__(self, event_log: EventLog):
        self.event_log = event_log
        self.root = None
        self.nodes = (
            []
        )  # TODO: Think about data structure for different types of nodes, dictionary sufficient?
        self.edges = []  # Format: (node1, node2)

    def find_exclusive_choice_cut(self, dfg: DirectlyFollowsGraph):
        # Convert the graph to undirected
        undirected = dfg.convert_to_undirected()
        # Find connected components
        cuts = undirected.find_components()

        print("Exclusive choice cuts: ", cuts)
        return None if len(cuts) == 1 else cuts
    
    def find_sequence_cut(self, dfg: DirectlyFollowsGraph):
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
        merged_nodes = list({node for pair in unreachable_pairs for node in pair if node in remaining_nodes})
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

        print("Sequence cuts: ", sorted_cuts)
        return None if len(sorted_cuts) == 1 else sorted_cuts

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
                if node1 != node2 and node2 not in updated_graph.graph[node1] and (node1, node2) not in removed_edges:
                    updated_graph.graph[node1].append(node2)

        # Remove dual edges to be removed
        for edge in removed_edges:
            updated_graph.graph[edge[0]].remove(edge[1])

        cuts = updated_graph.find_components()

        print("Parallel cuts: ", cuts)
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
        print("Components: ", components)

        # Find valid loop bodies

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
            for component in components:
                for neighbor in dfg.get_neighbors(node):
                    if neighbor in component and neighbor not in do_body:
                        # Merge with do-body
                        do_body.update(component)
                        components -= {component}

        # Find (invalid) components that are connected to non-start nodes in do-body
        for component in components:
            for node in component:
                for neighbor in dfg.get_neighbors(node):
                    if neighbor in do_body and neighbor not in start_nodes:
                        # Merge with do-body
                        do_body.update(component)
                        components -= {component}

        # Check connectivity completeness to start nodes
        for node in connected_nodes_start:
            for start_node in start_nodes:
                if start_node not in dfg.get_neighbors(node):
                    components_connected_to_start_nodes = {c for c in components_connected_to_start_nodes if node not in c}
                    break

        # Check reachability completeness from end nodes
        for node in connected_nodes_end:
            for end_node in end_nodes:
                if not node in dfg.get_neighbors(end_node):
                    components_connected_from_end_nodes = {c for c in components_connected_from_end_nodes if node not in c}
                    break

        # Remove components that are not connected to start nodes in do-body
        invalid_components = components - (components_connected_to_start_nodes.union(components_connected_from_end_nodes))
        loop_bodies = components - invalid_components

        # Merge invalid components to do-body
        for component in (invalid_components):
            do_body.update(component)

        # Add valid do-bodies to cuts
        cuts.append(list(do_body)) # Do-body has to be the first element
        for loop_body in loop_bodies:
            cuts.append(list(loop_body))

        # We need at least to components for a valid loop cut
        print("Loop cuts: ", cuts)
        return None if len(cuts) < 2 else cuts

    def exclusive_choice_split(self, event_log, cuts):
        # TODO
        pass

    def sequence_split(self, cuts):
        cuts = [set(cut) for cut in cuts]
        splits = []

        for trace in self.event_log.traces:
            trace_split = []
            cut_index = 0
            split = []

            for activity in trace:
                if activity in cuts[cut_index]:
                    split.append(activity)
                else:
                    if split:
                        trace_split.append(''.join(split))
                        split = []
                    cut_index = (cut_index + 1) % len(cuts)
                    split.append(activity)
            
            if split:
                trace_split.append(''.join(split))
            
            splits.append(trace_split)

        print("Sequence splits: ", splits)
        return split

    def parallel_split(self, cuts):
        # TODO
        pass

    def loop_split(self, cuts):
        # TODO
        pass

    def construct_process_tree(self):
        if "" in self.event_log.traces:
            return None
        for trace in self.event_log.traces:
            if len(trace) == 1:
                return None
            
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
                return (operator, splits)
            
        return None
    
class InductiveMiner():
    def __init__(self):
        pass

    def mine_process_model(self, event_log):
        # Step 1: Construct Directly-Follows Graph (DFG)
        dfg = DirectlyFollowsGraph(event_log)

        dfg.construct_dfg()
        dfg.print_graph()

        # Step 2: Construct Process Tree from DFG
        process_tree = ProcessTree(event_log)
        process_tree.construct_process_tree()


        return process_tree


if __name__ == "__main__":
    # event_log = EventLog.from_file("../data/log_from_paper.txt")
    # event_log.load_from_file()
    # event_log = EventLog.from_traces({'abcdfedfghabc': 3, 
    #                                   'abcdfeghabc': 2, 
    #                                   'abcijijkabc': 1})
    event_log = EventLog.from_traces({'abcd': 1, 'acbd':2})
    inductive_miner = InductiveMiner()
    process_tree = inductive_miner.mine_process_model(event_log)

    # print(process_tree)
