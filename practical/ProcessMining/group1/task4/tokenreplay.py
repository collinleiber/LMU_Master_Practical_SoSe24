class TokenReplay:
    """
    Token Replay algorithm implementation to check conformance of an event log with a Petri net model.

    Attributes:
        net: Dictionary representing the Petri net with input and output places for each transition.
        initial_marking: Dictionary representing the initial marking of the Petri net.
        final_marking: Dictionary representing the final marking of the Petri net.
        marking: Dictionary representing the current marking of the Petri net during replay.
        produced_tokens: Counter for the total number of produced tokens during replay.
        consumed_tokens: Counter for the total number of consumed tokens during replay.
        missing_tokens: Counter for the total number of missing tokens during replay.
        remaining_tokens: Counter for the total number of remaining tokens after replay.
    """

    def __init__(self, net, initial_marking, final_marking):
        """
        Initialize the TokenReplay class with a Petri net, initial marking, and final marking.

        Parameters:
            net: Dictionary representing the Petri net with input and output places for each transition.
            initial_marking: Dictionary representing the initial marking of the Petri net.
            final_marking: Dictionary representing the final marking of the Petri net.
        """
        self.net = net
        self.initial_marking = initial_marking
        self.final_marking = final_marking
        self.marking = initial_marking.copy()
        self.produced_tokens = 0
        self.consumed_tokens = 0
        self.missing_tokens = 0
        self.remaining_tokens = 0

    def replay_trace(self, trace):
        """
        Replay a single trace (sequence of events) through the Petri net.

        Parameters:
            trace: List of events representing a trace from the event log.
        """
        self.marking = self.initial_marking.copy()
        for event in trace:
            self.process_event(event)

    def process_event(self, event):
        """
        Process a single event by attempting to fire the corresponding transition in the Petri net.

        Parameters:
            event: The event to be processed.
        """
        if event == 'tau':
            return
        if self.can_fire(event):
            self.fire(event)
        else:
            self.missing_tokens += 1

    def can_fire(self, event):
        """
        Check if the transition corresponding to the event can be fired.

        Parameters:
            event: The event to be checked.

        Returns:
            bool: True if the transition can be fired, False otherwise.
        """
        for place in self.net['input'][event]:
            if self.marking[place] <= 0:
                return False
        return True

    def fire(self, event):
        """
        Fire the transition corresponding to the event, updating the marking.

        Parameters:
            event: The event to be fired.
        """
        for place in self.net['input'][event]:
            self.marking[place] -= 1
            self.consumed_tokens += 1
        for place in self.net['output'][event]:
            self.marking[place] += 1
            self.produced_tokens += 1

    def calculate_remaining_tokens(self):
        """
        Calculate the total number of remaining tokens in the Petri net after replay.
        """
        self.remaining_tokens = sum(self.marking.values())

    def run(self, log):
        """
        Run the Token Replay algorithm on an entire event log.

        Parameters:
            log: List of traces, where each trace is a list of events.

        Returns:
            dict: A dictionary containing the counts of produced, consumed, missing, and remaining tokens.
        """
        for trace in log:
            self.replay_trace(trace)
        self.calculate_remaining_tokens()
        return {
            'produced_tokens': self.produced_tokens,
            'consumed_tokens': self.consumed_tokens,
            'missing_tokens': self.missing_tokens,
            'remaining_tokens': self.remaining_tokens
        }
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