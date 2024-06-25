
from pm4py.visualization.dfg import visualizer as dfg_visualizer
from practical.ProcessMining.group2.inductive_miner.src.algo import DirectlyFollowsGraph, EventLog



#TODO add additional event logs and tests



l1 = EventLog({'abcd': 3,   'acbd' : 4, 'abcefbcd': 2,  'abcefcbd' : 1, 'acbefbcd' : 2, 'acbefbcefcbd' : 1})
l2 = EventLog({'bc': 3,   'cb' : 4, 'bcefbc': 2,  'bcefcb' : 1, 'cbefbc' : 2, 'cbefbcefcb' : 1})

dfg1 = DirectlyFollowsGraph(l1)
dfg1.construct_dfg()

graph = dfg1.graph

# Create a DFG
dfg = {}
for node, edges in graph.items():
    for edge in edges:
        if (node, edge) in dfg:
            dfg[(node, edge)] += 1
        else:
            dfg[(node, edge)] = 1

# Visualize the DFG
gviz = dfg_visualizer.apply(dfg, activities_count=None, parameters={"format": "png"})

dfg_visualizer.view(gviz) 


