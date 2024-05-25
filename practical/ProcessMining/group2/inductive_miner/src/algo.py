class EventLog:
    def __init__(self, event_log):
        self.event_log = event_log
        # TODO - which properties for a log?

    def load_from_file(self, file_path): # TODO - what is the format of the log file?
        pass

    # TODO - other input methods?

class DirectlyFollowsGraph:
    def __init__(self):
        # TODO - better data structures for nodes and edges?
        self.nodes = [] # Format: "node_id"
        self.edges = [] # Format: (node1, node2)

    def construct_dfg(self, event_log):
        pass

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
        dfg = DirectlyFollowsGraph()
        dfg.construct_dfg(event_log)

        # Step 2: Construct Process Tree from DFG
        process_tree = ProcessTree()
        process_tree.construct_process_tree(dfg)

        # Step 3: Refine and simplify the process tree
        process_tree.refine()

        return process_tree

if __name__ == "__main__":
    event_log = EventLog()
    event_log.load_from_file("../data/log_from_paper.txt")
    inductive_miner = InductiveMiner()
    process_tree = inductive_miner.mine_process_model(event_log)

    print(process_tree)
