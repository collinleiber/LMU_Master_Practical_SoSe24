import pm4py
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.algo.discovery.alpha import algorithm as alpha_miner
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
from pm4py.objects.conversion.process_tree import converter as pt_converter
import pandas as pd
from tokenreplay import TokenReplay
from comparison import ModelComparator

file_path = '../shared/example_files/DomesticDeclarations_afterDC.csv'
# file_path = '../shared/example_files/common-example.csv'

# Load the CSV file into a pandas DataFrame
log = pd.read_csv(file_path, sep=';')
log['timestamp'] = pd.to_datetime(log['timestamp'], utc=True)
event_log = pm4py.format_dataframe(log, case_id='case_id', activity_key='activity', timestamp_key='timestamp')
event_log = log_converter.apply(event_log)

# Alpha Miner
alpha_net, alpha_initial_marking, alpha_final_marking = alpha_miner.apply(event_log)
alpha_token_replay = TokenReplay(event_log, alpha_net, alpha_initial_marking, alpha_final_marking, "Alpha Miner")

# Inductive Miner
inductive_tree = inductive_miner.apply(event_log)
inductive_net, inductive_initial_marking, inductive_final_marking = pt_converter.apply(inductive_tree)
inductive_token_replay = TokenReplay(event_log, inductive_net, inductive_initial_marking, inductive_final_marking, "Inductive Miner")

# Inductive Miner Infrequent
inductive_infreq_tree = inductive_miner.apply(event_log, variant=inductive_miner.Variants.IMf)
inductive_infreq_net, inductive_infreq_initial_marking, inductive_infreq_final_marking = pt_converter.apply(inductive_infreq_tree)
inductive_infreq_token_replay = TokenReplay(event_log, inductive_infreq_net, inductive_infreq_initial_marking, inductive_infreq_final_marking, "Inductive Miner Infrequent")


# Heuristic Miner
heuristics_net, heuristics_initial_marking, heuristics_final_marking = heuristics_miner.apply(event_log)
heuristics_token_replay = TokenReplay(event_log, heuristics_net, heuristics_initial_marking,
                                      heuristics_final_marking, "Heuristic Miner")

# list of TokenReplay instances
model_list = [alpha_token_replay, inductive_token_replay, inductive_infreq_token_replay, heuristics_token_replay]

# Instantiate
comparator = ModelComparator(model_list)

# Pareto optimal models
pareto_optimal_models = comparator.run()

# Values of all models
all_models_values = comparator.get_models_values()

# Matrix of all models values
df = pd.DataFrame.from_dict(all_models_values, orient='index')
print(df)