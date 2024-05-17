from sortedcontainers import SortedSet, SortedDict
from itertools import chain
import numpy as np
from graphviz import Digraph


# Alpha Miner plus class
class AlphaMinerplus:
    def __init__(self, traces):
        self.traces = traces  # Traces from an event log
        self.transitions = SortedSet()
        self.initial_transitions = SortedSet()
        self.final_transitions = SortedSet()
        self.relations = SortedDict()  # Dictionary to keep of track of the relations ->, #, <-, ||
        self.pairs = []
        self.maximal_pairs = []
        self.places = []  # Set of places between maximal pairs
        self.length_one_loops = None

        # This corresponds to step 3 of the alpha plus algorithm:  3) T' := T \ L1L
        self.log_without_length_one_loops = None
        self.F_L1L = None
        self.W_minusL1L = SortedDict()

    def get_transitions(self) -> SortedSet:
        # Sets all transitions for the current petri net
        self.transitions = set(chain.from_iterable(self.traces.values()))

    def get_initial_transitions(self) -> SortedSet:
        # For each trace, get the first activity and add it to the set of initial transitions
        for index, trace in enumerate(self.traces.values(), start=1):
            self.initial_transitions.add(trace[0])

    def get_final_transitions(self) -> SortedSet:
        # For each trace get the last activity and add it to the set of final transitions
        for index, trace in enumerate(self.traces.values(), start=1):
            self.final_transitions.add(trace[len(trace) - 1])

    def get_footprint(self) -> np.ndarray:
        # Step 1: remove duplicate traces
        traces_without_duplicates = SortedSet()
        for trace in self.traces.values():
            traces_without_duplicates.add("".join(trace))

        # Step 2: Due to loop completeness, (definition 3.3: Ordering relations capturing length 2 loops) we need to examine all possible
        # firing sequences that have the form: aba with  t_i-1 = a ;   t_i = b   ;  t_i+1 = a
        # Extract relations between each transitions
        # generate Footprint

        for transition_1 in self.transitions:
            self.relations[transition_1] = SortedDict()
            for transition_2 in self.transitions:
                two_element_transitions = transition_1 + transition_2
                three_element_transition_1 = transition_2 + transition_1 + transition_2
                # eg aba, we also need bab
                three_element_transition_2 = transition_1 + transition_2 + transition_1

                all_relations = None
                for trace in traces_without_duplicates:
                    if all_relations is None:
                        if trace.find(two_element_transitions) >= 0:
                            all_relations = "->"
                        elif trace.find(two_element_transitions[::-1]) >= 0:
                            all_relations = "<-"
                    else:
                        if trace.find(two_element_transitions) >= 0:
                            if all_relations == "<-":
                                if (trace.find(three_element_transition_1) <= 0
                                    and trace.find(three_element_transition_2) <= 0):
                                    all_relations = "||"
                        elif trace.find(two_element_transitions[::-1]) >= 0:
                            if all_relations == "->":
                                if (trace.find(three_element_transition_1) <= 0
                                    and trace.find(three_element_transition_2) <= 0):
                                    all_relations = "||"

                if all_relations is None:
                    all_relations = "#"
                self.relations[transition_1][transition_2] = all_relations

    def get_pairs(self):
        # Generate pairs of activities, same procedure as the regular alpha miner
        # There must not be a relation between the activities
        # additinally the activities in the set have to be direcly successed by each other

        # Find pairs (A, B) of sets of activities such that every element a ∈ A and every element b ∈ B are
        # causally related (i.e. a →L b), all elements in A are independent (a1#a2), and all elements in B
        # are independent (b1#Lb2) as well

        choice_pairs = []

        causality_pairs = []

        # Extract all possible pairs of activities with the causality relation
        for activity1, relations1 in self.relations.items():
            for activity2, relation in relations1.items():
                if relation == "->":
                    causality_pairs.append((activity1, activity2))
                if relation == "#":
                    if activity1 == activity2:
                        choice_pairs.append((activity1,))
                    else:
                        choice_pairs.append((activity1, activity2))

        pairs = causality_pairs

        # We need to check the combinations too, while checking if the activities are independent in their respective sets

        i = 0  # TODO - there should be a more pythonic way to do this
        j = len(choice_pairs)

        while i < j:
            current_choice_pair = choice_pairs[i]
            for pair in choice_pairs:
                union = True
                if (len(SortedSet(current_choice_pair).intersection(SortedSet(pair)))
                    != 0):
                    for e1 in pair:
                        if not union:
                            break
                        for e2 in current_choice_pair:
                            if (self.relations[e1][e2] != "#"):  # check if the activities are independent
                                union = False
                                break
                    if union:  # if not independent, create a new pair which is the union of the 2 activities eg  [a,d]  +  [a, f] -> [a, d, f]
                        new_pair = SortedSet(current_choice_pair) | SortedSet(pair)
                        if tuple(new_pair) not in choice_pairs:
                            choice_pairs.append(tuple(new_pair))
                            j = j + 1
            i = i + 1

        # Union
        for pair_choices1 in choice_pairs:
            for pair_choices2 in choice_pairs:
                relation_between_pair = None
                makePair = True

                intersection = SortedSet(pair_choices1).intersection(pair_choices2)
                pair_choices2 = SortedSet(pair_choices2)

                if len(intersection) != 0:
                    # remove intersection terms in the second pair
                    for term in intersection:
                        pair_choices2.discard(term)

                if len(pair_choices2) == 0:
                    continue
                pair_choices2 = tuple(pair_choices2)

                for activity1 in pair_choices1:
                    if not makePair:
                        break
                    for activity2 in pair_choices2:
                        relation = self.relations[activity1][activity2]
                        if (relation_between_pair is not None
                            and relation_between_pair != relation):
                            makePair = False
                            break
                        else:
                            relation_between_pair = relation
                        if relation != "->":
                            makePair = False
                            break
                if makePair is True:
                    if relation_between_pair == "->":
                        new_pair = (pair_choices1, pair_choices2)
                    else:
                        new_pair = (pair_choices2, pair_choices1) # TODO- include in tests
                    pairs.append(new_pair)

        self.pairs = pairs

    def get_maximal_pairs(self):
        # Set of paired activities that are maximal
        pos1 = 0
        pair_appended = []
        maximal_pairs = []
        for pair1 in self.pairs:
            append_flag = True
            # flatten pair 1
            flat_pair1 = []
            for pair_element in pair1:
                for e in pair_element:
                    flat_pair1.append(e)

            pos2 = 0
            for pair2 in self.pairs:
                if pos1 != pos2:
                    flat_pair2 = []
                    for pair_element in pair2:
                        for e in pair_element:
                            flat_pair2.append(e)
                    # flat the pair 1

                    # flat the pair 2
                    # check if pair1 issubset of pair 2 or pair 2 is subset of 1
                    if (SortedSet(flat_pair1).issubset(flat_pair2) 
                    and SortedSet(flat_pair1) != SortedSet(flat_pair2)):
                        append_flag = False
                pos2 = pos2 + 1

            if append_flag is True:

                if SortedSet(flat_pair1) not in pair_appended:
                    maximal_pairs.append(pair1)
                    pair_appended.append(SortedSet(flat_pair1))
            pos1 = pos1 + 1

        self.maximal_pairs = maximal_pairs

    def get_length_one_loops(self) -> SortedSet:
        # extract length one loop
        self.length_one_loops = SortedSet()
        self.get_transitions()
        # compute footprint matrix and extract all transitions that have a causality relation with itself, eg: aa, bb etc.

        self.get_footprint()

        for transition in self.transitions:
            if self.relations[transition][transition] == "->":
                self.length_one_loops.add(transition) # TODO- include in tests

    def remove_length_one_loops(self) -> SortedSet:
        # This function corresponds to step 3 of the alpha plus algorithm:  3) T' := T \ L1L
        self.log_without_length_one_loops = self.transitions.difference(self.length_one_loops)

    def get_FL1L(self):
        # (You need T' for this )
        # Step 5 of the algorithm:
        # For each t ∈ L1L do:
        # (a) A = {a ∈ T'  | a >W t}
        # (b) B = {b ∈ T'  | t >W a}
        # (c) FL1L := FL1L ∪ {(t, p(A\B,B\A)),(p(A\B,B\A),t)}

        self.F_L1L = SortedSet()
        place_counter = 1
        self.get_transitions()
        for transition1 in self.length_one_loops: # TODO- include in tests
            A = SortedSet()
            B = SortedSet()
            for transition2 in self.log_without_length_one_loops:
                if self.relations[transition2][transition1] == "->":
                    A.add(transition2)
                if self.relations[transition1][transition2] == "->":
                    B.add(transition2)

            place = "p" + str(place_counter)
            for _ in A.difference(B):
                # Add input places
                transition_place = (transition1, place)
                self.F_L1L.add(transition_place)
            for _ in B.difference(A):
                # Add output place
                transition_place = (place, transition1)
                self.F_L1L.add(transition_place)
            place_counter += 1

    # We want the difference between 2 lists in generate_W_minus_L1L
    def diff(self, first, second) -> list:
        second = set(second)
        return [item for item in first if item not in second]

    def generate_W_minus_L1L(self):
        # W_minusL1L
        length_one_loops = self.length_one_loops
        for trace_key, trace in self.traces.items():
            trace_pr = trace
            trace_pr = self.diff(trace_pr, length_one_loops)
            self.W_minusL1L[trace_key] = trace_pr

    def add_places(self):
        # connect the initial transition with the first place, after the for loop we connect the last transition to the last place
        place_counter = 0
        self.places.append(
            ((), "Place_" + str(place_counter), self.initial_transitions)
        )
        place_counter = 1
        for pair in self.maximal_pairs:
            self.places.append((pair[0], "Place_" + str(place_counter), pair[1]))
            place_counter = place_counter + 1
        self.places.append(
            (self.final_transitions, ("Place_" + str(place_counter)), ())
        )

    def visualize(self):
        dot = Digraph()

        for transition in self.places:
            input_places, transition_name, output_places = (
                transition[0],
                transition[1],
                transition[2],
            )

            for input_place in input_places:
                dot.node(str(input_place), shape='circle')
                dot.edge(str(input_place), str(transition_name))
            dot.node(str(transition_name), shape='square')
            for output_place in output_places:
                dot.node(str(output_place), shape='circle')
                dot.edge(str(transition_name), str(output_place))

        dot.render('petri_net', format='png', cleanup=True)
