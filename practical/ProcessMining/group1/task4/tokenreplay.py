import pm4py


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

        self.fitness, self.simplicity, self.precision, self.generalization = self._calculate_pm4py_dimensions(
            log, net, initial_marking, final_marking)
        # Override / add dimension values of own implementation
        # self.fitness = self.calculate_fitness()

    def replay_trace(self, trace):
        """
        Replay a single trace (sequence of events) through the Petri net.

        Parameters:
            trace: List of events representing a trace from the event log.
        """
        self.marking = self.initial_marking.copy()
        for event in trace:
            self._process_event(event)

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
        else:
            self.missing_tokens += 1

    def _can_fire(self, event):
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

    def _fire(self, event):
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

    def _calculate_remaining_tokens(self):
        """
        Calculate the total number of remaining tokens in the Petri net after replay.
        """
        self.remaining_tokens = sum(self.marking.values())

    def run(self, log=None):
        """
        Run the Token Replay algorithm on an entire event log.

        Parameters:
            log: List of traces, where each trace is a list of events.

        Returns:
            dict: A dictionary containing the counts of produced, consumed, missing, and remaining tokens.
        """
        if not log:
            log = self.log
        for trace in log:
            self.replay_trace(trace)
        self._calculate_remaining_tokens()
        return {
            'produced_tokens': self.produced_tokens,
            'consumed_tokens': self.consumed_tokens,
            'missing_tokens': self.missing_tokens,
            'remaining_tokens': self.remaining_tokens
        }

    def get_fitness(self) -> float:
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

        fitness = (0.5 * (1 - (self.missing_tokens / self.consumed_tokens)) +
                   0.5 * (1 - (self.remaining_tokens / self.produced_tokens)))
        return fitness

    def get_simplicity(self):
        pass

    def _calculate_pm4py_dimensions(self, log, net, im, fm):
        fitness = pm4py.conformance.fitness_token_based_replay(log, net, im, fm)
        simplicity = pm4py.analysis.simplicity_petri_net(net, im, fm)
        precision = pm4py.conformance.precision_token_based_replay(log, net, im, fm)
        generalization = pm4py.conformance.generalization_tbr(log, net, im, fm)

        return fitness.get("log_fitness"), simplicity, precision, generalization
