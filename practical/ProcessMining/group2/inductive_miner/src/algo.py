class EventLog:
    def __init__(self, file_path):
        self.file_path = file_path
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
        self.nodes = {} # Format: "node_id" : [children]

    def construct_dfg(self):
        for trace in self.event_log.traces.keys():
            for i in range(len(trace) - 1):
                current_activity = trace[i]
                next_activity = trace[i + 1]
                if current_activity not in self.nodes:
                    self.nodes[current_activity] = []
                if next_activity not in self.nodes[current_activity]:
                    self.nodes[current_activity].append(next_activity)

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
    event_log.load_from_file()
    inductive_miner = InductiveMiner()
    process_tree = inductive_miner.mine_process_model(event_log)

    print(process_tree)
