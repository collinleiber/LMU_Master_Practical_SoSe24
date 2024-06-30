from practical.ProcessMining.group1.shared.visualizer import Visualizer
from practical.ProcessMining.group1.shared import utils
from practical.ProcessMining.group1.task2.alphaminer import AlphaMiner
from tokenreplay import TokenReplay
from comparison import ModelComparator

import pandas as pd
import pm4py

from pm4py.algo.discovery.alpha import algorithm as alpha_miner
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
from pm4py.algo.conformance.tokenreplay import algorithm as token_replay

from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.conversion.process_tree import converter as pt_to_petri_converter

from pm4py.visualization.petri_net import visualizer as pn_vis
from pm4py.visualization.process_tree import visualizer as pt_vis


# ================== logs ================

BASE = utils.SAMPLES_PATH

file_path_real = BASE / 'DomesticDeclarations_cleansed.csv'
file_path_common = BASE / 'common-example.csv'
file_path_limitation = BASE / 'limitation-example1.csv'

text_logs = BASE / 'simple_event_logs_modified.txt'

def get_log(file_path=file_path_real):
    return utils.import_csv(file_path)


log = get_log()
event_log = pm4py.format_dataframe(log, case_id='case_id', activity_key='activity', timestamp_key='timestamp')
event_log = log_converter.apply(event_log)

# logs = read_txt_test_logs("../shared/example_files/simple_event_logs_modified.txt")
# log = logs['L1']
# log = event_log_to_dataframe(log)
# log = pm4py.format_dataframe(log, case_id='case_id', activity_key='activity', timestamp_key='timestamp')
# event_log = log_converter.apply(log)


# ================== Miners ================

# # alpha Miner
alpha_net, alpha_initial_marking, alpha_final_marking = alpha_miner.apply(event_log)
gviz_alpha = pn_vis.apply(alpha_net, alpha_initial_marking, alpha_final_marking)
pm4py.visualization.petri_net.visualizer.view(gviz_alpha)

# replayed_traces = token_replay.apply(log, alpha_net, alpha_initial_marking, alpha_initial_marking)
# places = alpha_net.places
# vizard.display(gviz_alpha)
# graph_fitness = vizard.get_petri_net(alpha_net, alpha_initial_marking, alpha_initial_marking, replayed_traces)
# vizard.display(graph_fitness)

# Heuristic Miner
heuristics_net, heuristics_initial_marking, heuristics_final_marking = heuristics_miner.apply(event_log)
gviz_heuristics = pn_vis.apply(heuristics_net, heuristics_initial_marking, heuristics_final_marking)
pm4py.visualization.petri_net.visualizer.view(gviz_heuristics)

# IM
process_tree_IM = inductive_miner.apply(log, variant=inductive_miner.Variants.IM)
# gviz_tree_IM = pt_vis.apply(process_tree_IM)
# pt_vis.view(gviz_tree_IM)
net, initial_marking, final_marking = pt_to_petri_converter.apply(process_tree_IM)
gviz_petri_IM = pn_vis.apply(net, initial_marking, final_marking)
pn_vis.view(gviz_petri_IM)

parameters = {
    'noise_threshold': 0.2
}
# IMf
process_tree_IMf = inductive_miner.apply(log, parameters=parameters,variant=inductive_miner.Variants.IMf)
# gviz_tree_IMf = pt_visualizer.apply(process_tree_IMf)
# pt_visualizer.view(gviz_tree_IMf)
net, initial_marking, final_marking = pt_to_petri_converter.apply(process_tree_IMf)
gviz_petri_IMf = pn_vis.apply(net, initial_marking, final_marking)
pn_vis.view(gviz_petri_IMf)

# ================== Token Replay ================

# Alpha Miner
alpha_net, alpha_initial_marking, alpha_final_marking = alpha_miner.apply(event_log)
alpha_token_replay = TokenReplay(event_log, alpha_net, alpha_initial_marking, alpha_final_marking, "Alpha Miner")
# print('al_token_replay ======', alpha_token_replay)
vizard = Visualizer()
places = alpha_net.places
print('places:', places)
tokens = alpha_token_replay.get_tokens()
print('tokens ==========', tokens)
graph_fitness = vizard.build_petri_net(alpha_net, alpha_initial_marking, alpha_final_marking, tokens)
print('graph_fitness ===', graph_fitness)
vizard.save(graph_fitness,'graph_fitness')

# Inductive Miner
inductive_tree = inductive_miner.apply(event_log)
inductive_net, inductive_initial_marking, inductive_final_marking = pt_to_petri_converter.apply(inductive_tree)
inductive_token_replay = TokenReplay(event_log, inductive_net, inductive_initial_marking, inductive_final_marking, "Inductive Miner")

# Inductive Miner Infrequent
inductive_inf_tree = inductive_miner.apply(event_log, variant=inductive_miner.Variants.IMf)
inductive_inf_net, inductive_inf_initial_marking, inductive_inf_final_marking = pt_to_petri_converter.apply(inductive_inf_tree)
inductive_inf_token_replay = TokenReplay(event_log, inductive_inf_net, inductive_inf_initial_marking, inductive_inf_final_marking, "Inductive Miner Infrequent")


# Heuristic Miner
heuristics_net, heuristics_initial_marking, heuristics_final_marking = heuristics_miner.apply(event_log)
heuristics_token_replay = TokenReplay(event_log, heuristics_net, heuristics_initial_marking,
                                      heuristics_final_marking, "Heuristic Miner")

# list of TokenReplay instances
model_list = [alpha_token_replay, inductive_token_replay, inductive_inf_token_replay, heuristics_token_replay]

# Instantiate
comparator = ModelComparator(model_list)

# Pareto optimal models
pareto_optimal_models = comparator.run()

# Values of all models
all_models_values = comparator.get_models_values()

# Matrix of all models values
df = pd.DataFrame.from_dict(all_models_values, orient='index')
print(df)

