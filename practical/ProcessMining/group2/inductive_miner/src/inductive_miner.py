import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from src.graph import *
from src.event_log import *
from src.directly_follows_graph import *
from src.process_tree import *


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
    