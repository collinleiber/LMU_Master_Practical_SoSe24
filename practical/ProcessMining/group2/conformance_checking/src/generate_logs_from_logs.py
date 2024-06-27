from models_from_pm4py import get_model_from_pm4py, AlgoPm4Py
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from models_from_pm4py import get_model_from_pm4py
from replay import get_traces_with_replay

log_name = "L5"
input_log_path = "./../InputLogs/{}.csv".format(log_name)
output_log_path = "./../OutputLogs/{}.csv".format(log_name)

net, start, end = get_model_from_pm4py(input_log_path, AlgoPm4Py.ALPHAPLUS)
# gviz = pn_visualizer.apply(net, start, end)
# pn_visualizer.view(gviz)

traces = get_traces_with_replay(net, start, end, len(net.transitions) * 2)
with open(output_log_path, 'w') as file:
    for id, trace in enumerate(traces):
        for activity in trace:
            file.write("{},{}".format(id + 1, activity))
            file.write('\n')
