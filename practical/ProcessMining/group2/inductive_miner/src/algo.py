import os

script_dir = os.path.dirname(os.path.abspath(__file__))

class EventLog:
    def __init__(self, relative_file_path):
        self.file_path = os.path.join(script_dir, relative_file_path)
        self.traces = {}

    def load_from_file(self): # TODO - what is the format of the log file?
        with open(self.file_path, 'r') as file:
            for line in file:
                trace = line.strip()
                if trace in self.traces:
                    self.traces[trace] += 1
                else:
                    self.traces[trace] = 1
        
    # TODO - other input methods?

class DirectlyFollowsGraph:
    def __init__(self, event_log: EventLog):
        self.event_log = event_log
        self.graph = {} # Adjacency list, format: "node_id" : [children]
        self.start_nodes = set()
        self.end_nodes = set()

    def construct_dfg(self):
        for trace in self.event_log.traces.keys():
            if trace:
                self.start_nodes.add(trace[0])
                self.end_nodes.add(trace[-1])

                for i in range(len(trace) - 1):
                    current_activity = trace[i]
                    next_activity = trace[i + 1]

                    if current_activity not in self.graph:
                        self.graph[current_activity] = []
                    if next_activity not in self.graph[current_activity]:
                        self.graph[current_activity].append(next_activity)
                    if next_activity not in self.graph:
                        self.graph[next_activity] = []

    # Debugging helper
    def print_graph(self):
        print("Graph: ", self.graph)
        print("Start nodes: ", self.start_nodes)
        print("End nodes: ", self.end_nodes)

class ProcessTree:
    def __init__(self):
        self.root = None
        self.nodes = [] # TODO: Think about data structure for different types of nodes, dictionary sufficient?
        self.edges = [] # Format: (node1, node2)

    def find_exclusive_choice_split(self, directly_follows_graph: DirectlyFollowsGraph):
        pass

    def find_sequence_split(self, directly_follows_graph: DirectlyFollowsGraph):
        pass

    def find_parallel_split(self, directly_follows_graph: DirectlyFollowsGraph):
        pass

    def find_loop_split(self, directly_follows_graph: DirectlyFollowsGraph):
        pass

    def construct_process_tree(self, directly_follows_graph: DirectlyFollowsGraph):
        pass

class InductiveMiner():
    def __init__(self):
        pass

    def mine_process_model(self, event_log):
        # Step 1: Construct Directly-Follows Graph (DFG)
        dfg = DirectlyFollowsGraph(event_log)
        dfg.construct_dfg()

        # Step 2: Construct Process Tree from DFG
        process_tree = ProcessTree()
        process_tree.construct_process_tree(dfg)

        return process_tree

if __name__ == "__main__":
    event_log = EventLog("../data/log_from_paper.txt")
    # event_log.load_from_file()
    event_log.traces = {'abcd': 3, 'acbd': 2, 'aed': 1}
    inductive_miner = InductiveMiner()
    process_tree = inductive_miner.mine_process_model(event_log)

    # print(process_tree)
