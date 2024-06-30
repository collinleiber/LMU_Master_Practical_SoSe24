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
        self.missing_tokens = 0
        self.remaining_tokens = 0
        self.missing_details = defaultdict(int)
        self.remaining_details = defaultdict(int)

        self.precision_numerator = 0
        self.precision_denominator = 0
        self.fitness, self.simplicity, self.precision, self.generalization = self._calculate_pm4py_dimensions(
            log, net, initial_marking, final_marking)
        # Override / add dimension values of own implementation
        # self.fitness = self.calculate_fitness()
        self.tokens = self._calculate_missing_remaining_tokens(log, net, initial_marking, final_marking)

    def replay_trace(self, trace):
        """
        Replay a single trace (sequence of events) through the Petri net.

        Parameters:
            trace: List of events representing a trace from the event log.

        Returns:
            dict: Results of token replay for the trace.
        """
        self.marking = self.initial_marking.copy()
        self.produced_tokens = 0
        self.consumed_tokens = 0
        self.missing_tokens = 0
        self.remaining_tokens = 0
        self.missing_details.clear()
        self.remaining_details.clear()

        for event in trace:
            self._process_event(event)

        self._calculate_remaining_tokens()
        return {
            'produced_tokens': self.produced_tokens,
            'consumed_tokens': self.consumed_tokens,
            'missing_tokens': self.missing_tokens,
            'remaining_tokens': self.remaining_tokens,
            'missing_details': dict(self.missing_details),
            'remaining_details': dict(self.remaining_details)
        }

    def _process_event(self, event):
        """
        Process a single event by attempting to fire the corresponding transition in the Petri net.

        Parameters:
            event: The event to be processed.
        """
        if event == 'tau':
            return
        if self._can_fire(event):
            self._fire(event)
            self.precision_numerator += 1  # Count of successfully fired transitions
        else:
            self.missing_tokens += 1
            transition = next((t for t in self.net.transitions if t.label == event), None)
            if transition:
                for arc in transition.in_arcs:
                    self.missing_details[(frozenset([arc.source.name]), frozenset([arc.target.name]))] += 1
        self.precision_denominator += 1  # Count of all attempted transitions

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

    def _fire(self, event):
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
            self.consumed_tokens += 1
        for arc in transition.out_arcs:
            self.marking[arc.target] += 1
            self.produced_tokens += 1

    def _calculate_remaining_tokens(self):
        """
        Calculate the total number of remaining tokens in the Petri net after replay.
        """
        for place, tokens in self.marking.items():
            if tokens > 0:
                self.remaining_tokens += tokens
                self.remaining_details[frozenset([place.name])] += tokens

    # for test
    def _calculate_missing_remaining_tokens(self, log, net, im, fm):
        replayed_traces = token_replay.apply(log, net, im, fm)
        # print('replayed_traces ====', replayed_traces)

        places = net.places

        missing = defaultdict(int)
        remaining = defaultdict(int)

        for trace in replayed_traces:
            if isinstance(trace['missing_tokens'], dict):
                for place, count in trace['missing_tokens'].items():
                    missing[place] += count
            elif isinstance(trace['missing_tokens'], int):
                for place in places:
                    missing[place] += trace['missing_tokens']

            if isinstance(trace['remaining_tokens'], dict):
                for place, count in trace['remaining_tokens'].items():
                    remaining[place] += count
            elif isinstance(trace['remaining_tokens'], int):
                for place in places:
                    remaining[place] += trace['remaining_tokens']

        return {
            'missing': missing,
            'remaining': remaining
        }

    def get_tokens(self):
        return self.tokens

    def run(self, log=None):
        """
        Run the Token Replay algorithm on an entire event log.

        Parameters:
            log: List of traces, where each trace is a list of events.

        Returns:
            list: A list of results for each trace.
        """
        self.produced_tokens = 0
        self.consumed_tokens = 0
        self.missing_tokens = 0
        self.remaining_tokens = 0
        self.precision_numerator = 0
        self.precision_denominator = 0

        if not log:
            log = self.log
        results = []
        for trace in log:
            result = self.replay_trace(trace)
            results.append(result)

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
        fitness = (0.5 * (1 - (self.missing_tokens / self.consumed_tokens)) +
                   0.5 * (1 - (self.remaining_tokens / self.produced_tokens)))
        return fitness

    def calculate_simplicity(self):
        pass

    def _calculate_pm4py_dimensions(self, log, net, im, fm):
        fitness = pm4py.conformance.fitness_token_based_replay(log, net, im, fm)
        simplicity = pm4py.analysis.simplicity_petri_net(net, im, fm)
        precision = pm4py.conformance.precision_token_based_replay(log, net, im, fm)
        generalization = pm4py.conformance.generalization_tbr(log, net, im, fm)

        return fitness.get("log_fitness"), simplicity, precision, generalization
