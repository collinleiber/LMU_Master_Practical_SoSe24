from collections import defaultdict

import pm4py


class TokenReplay:
    # Generic nets to provide the token replay algorithm for log2log, log2model, and model2model
    def __init__(self, event_log, model_net):
        self.event_log = event_log
        self.model_net = model_net
        self.fitness, self.simplicity = self.replay()

    def replay(self, traces):
        # Implement the token replay algorithm
        result = []
        for trace in traces:
            produced, consumed, missing, remaining = 0, 0, 0, 0

            # TODO replay the trace

            result.append((produced, consumed, missing, remaining))

        fitness = self.get_fitness(result)
        simplicity = self.get_simplicity()

    def get_fitness(self, token_results) -> float:
        kpi = defaultdict(int)

        for produced, consumed, missing, remaining in token_results:
            kpi['p'] += produced
            kpi['c'] += consumed
            kpi['m'] += missing
            kpi['r'] += remaining

        fitness = 0.5 * (1 - (kpi['m'] / (kpi['c']) + 0.5 * (kpi['r'] / kpi['p'])))
        return fitness

    def get_simplicity_pm4py(self, net, init_marking, final_marking):
        return pm4py.analysis.simplicity_petri_net(net, init_marking, final_marking)

    def get_precision_pm4py(self, net, init_marking, final_marking):
        pm4py.conformance.precision_token_based_replay()