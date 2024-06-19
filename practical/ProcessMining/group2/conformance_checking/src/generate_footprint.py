from sortedcontainers import SortedSet, SortedDict
from itertools import chain
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt
class Relations(Enum):
    BEFORE = '<-'
    SEQUENCE =  '->'   
    NOT_FOLLOWED = '#' 
    PARALLEL = '||'  



class Conformance_checking:
    def __init__(self, traces):
        self.traces = traces  # Traces from an event log
        self.transitions = SortedSet()
        
        self.relations = (
            SortedDict()
        )  # Dictionary to keep of track of the relations ->, #, <-, ||
       
        self.places = []  # Set of places between maximal pairs
      

    def get_transitions(self):
        # Sets all transitions for the current petri net
        self.transitions = set(chain.from_iterable(self.traces.values()))


    def get_footprint_regular_alpha_miner(self) -> np.ndarray:
        # Step 1: remove duplicate traces
        traces_without_duplicates = SortedSet()
        
        for trace in self.traces.values():
            traces_without_duplicates.add("".join(trace))
       
        # Extract relations between each transitions
        # generate Footprint

        for transition_1 in self.transitions:
            self.relations[transition_1] = SortedDict()
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

    def visualize_sorted_dict(self, sorted_dict):
        keys = list(sorted_dict.keys())
        size = len(keys)
        
        

        fig, ax = plt.subplots(figsize=(7, 7)) 
    
        ax.matshow([[0] * size] * size, cmap='Greys')  # Create an empty heatmap

        # Add gridlines
        ax.set_xticks([x - 0.5 for x in range(1, size)], minor=True)
        ax.set_yticks([y - 0.5 for y in range(1, size)], minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=5)
        for i, row_key in enumerate(keys):
            for j, col_key in enumerate(keys):
                ax.text(j, i, sorted_dict[row_key][col_key], ha='center', va='center', color='black', fontsize=38)

        ax.set_xticks(range(size))
        ax.set_yticks(range(size))
        ax.set_xticklabels(keys, fontsize=28)
        ax.set_yticklabels(keys, fontsize=28)
        ax.tick_params(axis='both', which='major', labelsize=28)
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(5)  # Adjust the linewidth for the outer border
        plt.title('Footprint Matrix', fontsize=22)
        plt.show()

#TODO add different conformance checking variations
#TODO implement comparison
#TODO add comments




