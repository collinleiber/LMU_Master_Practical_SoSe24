import os
import sys

from collections import defaultdict
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from src.graph import Graph
from src.event_log import EventLog



class DirectlyFollowsGraph(Graph):
    def __init__(self, event_log: EventLog) -> None:
        """
        Initialize DirectlyFollowsGraph object.

        :param event_log: EventLog object.
        """
        self.event_log = event_log
        self.graph = defaultdict(list)  # Adjacency list, format: "node_id" : [children]
        self.start_nodes = set()
        self.end_nodes = set()

    def construct_dfg(self) -> None:
        """Construct Directly Follows Graph."""
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

    def __str__(self) -> str:
        """Represent Directly Follows Graph as string."""
        return f"Directly Follows Graph: (\n\tGraph: {dict(self.graph)}\n\tStart nodes: {self.start_nodes}\n\tEnd nodes: {self.end_nodes}\n)"
