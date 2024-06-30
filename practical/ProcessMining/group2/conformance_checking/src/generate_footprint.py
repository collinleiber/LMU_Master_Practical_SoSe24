from itertools import chain
import numpy as np
from enum import Enum


class Relations(Enum):
    BEFORE = '<-'
    SEQUENCE = '->'
    NOT_FOLLOWED = '#'
    PARALLEL = '||'


class FootPrintMatrix:
    def __init__(self, log=None, relations=None):
        if log is not None:
            self.traces = self.convert_log_for_footprintmatrix(log)
        else:
            self.traces = []
        self.transitions = set()

        if relations is None:
            self.relations = {}  # Default to an empty dict
        else:
            self.relations = relations

        self.places = []

    def sort_fpm_rec(self, relations):
        sorted_dict = {}
        for key in sorted(relations.keys()):
            value = relations[key]
            if isinstance(value, dict):
                sorted_dict[key] = self.sort_fpm_rec(value)
            else:
                sorted_dict[key] = value
        return sorted_dict

    @classmethod
    def from_relations(cls, relations):
        return cls(relations=relations)

    def convert_log_for_footprintmatrix(self, log):
        traces = {}
        trace_num = 1

        for trace in log:
            activities = []
            for event in trace:
                activity_name = event['concept:name']
                activities.append(activity_name)

            traces[str(trace_num)] = activities
            trace_num += 1

        return traces

    def generate_transitions(self):
        # Sets all transitions for the current petri net
        self.transitions = set(chain.from_iterable(self.traces.values()))

    def generate_footprint(self) -> np.ndarray:
        print("Generating a Footprint Matrix!")
        # Step 0: generate transitions
        self.generate_transitions()
        # Step 1: remove duplicate traces
        traces_without_duplicates = set()

        for trace in self.traces.values():
            traces_without_duplicates.add("".join(trace))

        # Extract relations between each transitions
        # generate Footprint

        for transition_1 in self.transitions:
            self.relations[transition_1] = {}
            for transition_2 in self.transitions:
                two_element_transitions = transition_1 + transition_2

                all_relations = None
                for trace in traces_without_duplicates:

                    if trace.find(two_element_transitions) >= 0:
                        # print(two_element_transitions)
                        # all_relations = "->"
                        if all_relations == Relations.BEFORE.value:

                            all_relations = Relations.PARALLEL.value
                        else:
                            all_relations = Relations.SEQUENCE.value

                    if trace.find(two_element_transitions[::-1]) >= 0:

                        if all_relations == Relations.SEQUENCE.value:

                            all_relations = Relations.PARALLEL.value
                        else:
                            all_relations = Relations.BEFORE.value

                if all_relations == None:
                    all_relations = Relations.NOT_FOLLOWED.value
                self.relations[transition_1][transition_2] = all_relations

        self.relations = self.sort_fpm_rec(self.relations)
