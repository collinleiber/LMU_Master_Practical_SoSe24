import os
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
    def __init__(self):
        self.root = None
        self.nodes = (
            []
        )  # TODO: Think about data structure for different types of nodes, dictionary sufficient?
        self.edges = []  # Format: (node1, node2)

    def find_exclusive_choice_split(self, dfg: DirectlyFollowsGraph):
        # Convert the graph to undirected
        undirected = dfg.convert_to_undirected()
        # Find connected components
        cuts = undirected.find_components()

        return None if len(cuts) == 1 else cuts
    
    def find_sequence_split(self, dfg: DirectlyFollowsGraph):
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

        return None if len(cuts) == 1 else sorted_cuts

    def find_parallel_split(self, dfg: DirectlyFollowsGraph):
        # Mark edges to be removed
        edges = set(dfg.get_all_edges())
        removed_edges = set()
        for edge in edges:
            if edge[::-1] in edges:
                removed_edges.add(edge)

        # Insert edge where no edge exists between two nodes
        updated_graph = Graph(dfg.graph.copy())
        nodes = dfg.get_all_nodes()
        for node1 in nodes:
            for node2 in nodes:
                if node1 != node2 and node2 not in updated_graph.graph[node1] and (node1, node2) not in removed_edges:
                    updated_graph.graph[node1].append(node2)

        # Remove dual edges to be removed
        for edge in removed_edges:
            updated_graph.graph[edge[0]].remove(edge[1])

        cuts = updated_graph.find_components()

        return None if len(cuts) == 1 else cuts

    def find_loop_split(self, dfg: DirectlyFollowsGraph):
        pass

    def construct_process_tree(self, dfg: DirectlyFollowsGraph):
        # self.find_exclusive_choice_split(dfg)
        # self.find_sequence_split(dfg)
        self.find_parallel_split(dfg)
class InductiveMiner():
    def __init__(self):
        pass

    def mine_process_model(self, event_log):
        # Step 1: Construct Directly-Follows Graph (DFG)
        dfg = DirectlyFollowsGraph(event_log)

        dfg.construct_dfg()
        dfg.print_graph()

        # Step 2: Construct Process Tree from DFG
        process_tree = ProcessTree()
        process_tree.construct_process_tree(dfg)

        return process_tree


if __name__ == "__main__":
    # event_log = EventLog.from_file("../data/log_from_paper.txt")
    # event_log.load_from_file()
    event_log = EventLog.from_traces({'abcd': 3, 'acbd': 2, 'aed': 1})
    inductive_miner = InductiveMiner()
    process_tree = inductive_miner.mine_process_model(event_log)

    # print(process_tree)
