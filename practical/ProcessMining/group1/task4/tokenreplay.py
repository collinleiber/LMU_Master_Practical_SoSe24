from collections import defaultdict


class TokenReplay:
    """
    Token Replay algorithm implementation to check conformance of an event log with a Petri net model.
    """

    def __init__(self, log, net, initial_marking, final_marking, net_type):
        """
        Initialize the TokenReplay class with a Petri net, initial marking, and final marking.

        Parameters:
            log: Event log used for conformance checking with the given net as model.
            net: Petri net model to check conformance against.
            initial_marking: Initial marking of the Petri net.
            final_marking: Final marking of the Petri net.
            net_type: Description of the discovery method used.
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
        self.fitness = 0.0
        self.simplicity = 0.0
        self.precision = 0.0
        self.generalization = 0.0

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

        # Update fitness, simplicity, precision, generalization
        self.fitness = self.calculate_fitness()
        self.simplicity = self.calculate_simplicity()
        self.precision = self.calculate_precision()
        self.generalization = self.calculate_generalization()

        return results

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
        """
        Calculate the fitness of the model. Fitness is the measure of how well the model reproduces the observed behavior.

        Returns:
            float: The fitness value between 0 and 1.
        """
        if self.consumed_tokens == 0 or self.produced_tokens == 0:
            return 0.0
        fitness = (0.5 * (1 - (self.missing_tokens / self.consumed_tokens)) +
                   0.5 * (1 - (self.remaining_tokens / self.produced_tokens)))
        return fitness

    def calculate_simplicity(self):
        """
        Calculate the simplicity of the Petri net model. Simplicity measures the complexity of the model.

        Returns:
            float: The simplicity value.
        """
        transitions = len(self.net.transitions)
        places = len(self.net.places)
        arcs = sum(len(transition.in_arcs) + len(transition.out_arcs) for transition in self.net.transitions)
        simplicity = 1 / (transitions + places + arcs)
        return simplicity

    def calculate_precision(self):
        """
        Calculate the precision of the Petri net model. Precision measures the ratio of correctly fired transitions to attempted transitions.

        Returns:
            float: The precision value.
        """
        if self.precision_denominator == 0:
            return 0.0
        return self.precision_numerator / self.precision_denominator

    def calculate_generalization(self):
        """
        Calculate the generalization of the Petri net model. Generalization measures how well the model generalizes to unseen traces.

        Returns:
            float: The generalization value.
        """
        seen_traces = set(tuple(trace) for trace in self.log)
        all_possible_traces = self._generate_all_possible_traces()
        if len(all_possible_traces) == 0:
            return 1.0
        generalization = len(seen_traces) / len(all_possible_traces)
        return generalization

    def _generate_all_possible_traces(self):
        """
        Generate all possible traces for the given Petri net model.
        Showcase implementation, not feasible for large models.
        """
        return set(tuple(trace) for trace in self.log)

