from practical.ProcessMining.group2.inductive_miner.src.graph import *
from practical.ProcessMining.group2.inductive_miner.src.event_log import *
from practical.ProcessMining.group2.inductive_miner.src.directly_follows_graph import *
from practical.ProcessMining.group2.inductive_miner.src.process_tree import *


class InductiveMiner():
    def mine_process_model(self, event_log: EventLog) -> ProcessTree:
        """
        Mine process model using Inductive Miner.

        :param event_log: EventLog object.
        :return: ProcessTree object.
        """
        # Construct Directly-Follows Graph (DFG)
        dfg = DirectlyFollowsGraph(event_log)
        dfg.construct_dfg()

        # Construct Process Tree from DFG
        process_tree = ProcessTree(event_log)

        return process_tree
    