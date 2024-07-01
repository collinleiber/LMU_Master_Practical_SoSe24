from collections import defaultdict
import pm4py
from pm4py.algo.conformance.tokenreplay import algorithm as token_replay


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

    def __init__(self, log, net, initial_marking, final_marking, net_type):
        """
        Initialize the TokenReplay class with a Petri net, initial marking, and final marking.

        Parameters:
            log: Base event log used for conformance checking with given net as model.
            net: Dictionary representing the Petri net with input and output places for each transition.
            initial_marking: Dictionary representing the initial marking of the Petri net.
            final_marking: Dictionary representing the final marking of the Petri net.
            net_type: Descriptions which Discovery method has been used.
        """
        self.log = log
        self.net = net
        self.net_type = net_type
        self.initial_marking = initial_marking
        self.final_marking = final_marking
        self.marking = initial_marking.copy()
        self.produced_tokens = 0
        self.consumed_tokens = 0
        self.missing_tokens = defaultdict(int)
        self.remaining_tokens = defaultdict(int)

        self.fitness, self.simplicity, self.precision, self.generalization = self._calculate_pm4py_dimensions(
            log, net, initial_marking, final_marking)
        # Override / add dimension values of own implementation
        # self.fitness = self.calculate_fitness()

    def run(self, log=None):
        """
        Run the Token Replay algorithm on an entire event log.

        Parameters:
            log: List of traces, where each trace is a list of events.

        Returns:
            list: A list of results for each trace.
        """
        if not log:
            log = self.log
        for index, trace in enumerate(log):
            trace_result = self.replay_trace(trace)
            self.produced_tokens += trace_result['produced_tokens']
            self.produced_tokens += trace_result['consumed_tokens']
            self.missing_tokens |= trace_result['missing_tokens']
            self.remaining_tokens |= trace_result['remaining_tokens']

    def replay_trace(self, trace):
        """
        Replay a single trace (sequence of events) through the Petri net.

        Parameters:
            trace: List of events representing a trace from the event log.

        Returns:
            dict: Results of token replay for the trace.
        """
        self.marking = self.initial_marking.copy()
        produced_tokens = 0
        consumed_tokens = 0
        missing_tokens = defaultdict(int)

        def handle_unconformity():
            transition = next((t for t in self.net.transitions if t.label == event), None)
            if transition:
                for arc in transition.in_arcs:
                    missing_tokens[(arc.source, arc.target)] += 1

        def handle_tau():
            # Check the next event in the trace
            if i + 1 < len(trace) and self._can_fire(trace[i + 1]['activity']):
                # Produce and consume a token for the tau event
                self._fire(event, produced_tokens, consumed_tokens)

        for i, event in enumerate(trace):
            if 'activity' in event:
                event = event['activity']

                if event == 'tau':
                    handle_tau()
                elif self._can_fire(event):
                    self._fire(event, produced_tokens, consumed_tokens)
                else:
                    handle_unconformity()
            else:
                raise KeyError('Event Log structure not a dict with a key "activity"')

        return {
            'produced_tokens': produced_tokens,
            'consumed_tokens': consumed_tokens,
            'missing_tokens': missing_tokens,
            'remaining_tokens': self._calculate_remaining_tokens(),
        }

    def _can_fire(self, event):
        """
        Check if the transition corresponding to the event can be fired.

        Parameters:
            event: The event to be checked.

        Returns:
            bool: True if the transition can be fired, False otherwise.
        """
        transition = next((t for t in self.net.transitions if t.label == event), None)
        if transition is None:
            return False
        for arc in transition.in_arcs:
            if self.marking[arc.source] <= 0:
                return False
        return True

    def _fire(self, event, produced: int, consumed: int):
        """
        Fire the transition corresponding to the event, updating the marking.

        Parameters:
            event: The event to be fired.
        """
        transition = next((t for t in self.net.transitions if t.label == event), None)
        if transition is None:
            return
        for arc in transition.in_arcs:
            self.marking[arc.source] -= 1
            consumed += 1
        for arc in transition.out_arcs:
            self.marking[arc.target] += 1
            produced += 1

    def _calculate_remaining_tokens(self):
        """
        Calculate the total number of remaining tokens in the Petri net after replay.
        """
        for place, tokens in self.marking.items():
            if tokens > 0:
                self.remaining_tokens += tokens
                self.remaining_details[frozenset([place.name])] += tokens


    def get_unconformity_tokens(self):
        return {
            "missing": self.missing_tokens,
            "remaining": self.remaining_tokens
        }

    def get_discovery_type(self):
        return self.net_type

    def get_fitness(self):
        return self.fitness

    def get_simplicity(self):
        return self.simplicity

    def get_precision(self):
        return self.precision

    def get_generalization(self):
        return self.generalization

    def get_dimension_value(self, dimension: str):
        if dimension in ('f', 'fitness'):
            return self.get_fitness()
        elif dimension in ('s', 'simplicity'):
            return self.get_simplicity()
        elif dimension in ('p', 'precision'):
            return self.get_precision()
        elif dimension in ('g', 'generalization'):
            return self.get_generalization()
        else:
            return ValueError

    def calculate_fitness(self) -> float:
        total_missing = sum(val for val in self.missing_tokens.values())
        total_remaining = sum(val for val in self.remaining_tokens.values())

        fitness = (0.5 * (1 - (total_missing / self.consumed_tokens)) +
                   0.5 * (1 - (total_remaining / self.produced_tokens)))
        return fitness

    def calculate_simplicity(self):
        pass

    def _calculate_pm4py_dimensions(self, log, net, im, fm):
        fitness = pm4py.conformance.fitness_token_based_replay(log, net, im, fm)
        simplicity = pm4py.analysis.simplicity_petri_net(net, im, fm)
        precision = pm4py.conformance.precision_token_based_replay(log, net, im, fm)
        generalization = pm4py.conformance.generalization_tbr(log, net, im, fm)

        return fitness.get("log_fitness"), simplicity, precision, generalization
