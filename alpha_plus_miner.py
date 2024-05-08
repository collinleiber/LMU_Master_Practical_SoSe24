
from enum import Enum
from sortedcontainers import SortedSet, SortedDict
import numpy as np
import time
import pm4py
from pm4py.visualization.footprints import visualizer
#import graphviz


# Alpha Miner plus class
class AlphaMinerplus:

    def __init__(self,Traces):
        # Traces from an event log
        self.traces = Traces
        # set of transitions as a sorted set
        self.transitions = SortedSet()
        # set of initial transitions Ti
        self.initial_transitions = SortedSet()
        # set of final transitions To
        self.final_transitions = SortedSet()
        #dictionary to keep of track of the relations ->, #, <-, ||
        self.relations = SortedDict()


        
       # self.footprint_matrix_reshaped #= self.getFootprint()

    def getTransitions(self) -> SortedSet: 
        #returns all transitions for the current petri net
        for trace in self.traces.values():
            print("current trace: ", trace)
            for activity in trace:
                self.transitions.add(activity)
                print("current activity: " , activity)

       # print("getTransition returns: ", self.transitions)
        return self.transitions
    

    def getInitialTransitions(self) -> SortedSet:
        
        #For each trace get the first activity and add it to the set of initial transitions
        for index, trace in enumerate(self.traces.values(), start=1):
            print(f"Initial transition for trace {index}: {trace[0]}")
           
            self.initial_transitions.add(trace[0])
        return self.initial_transitions

    def getFinalTransitions(self) -> SortedSet:
        #For each trace get the last activity and add it to the set of final transitions
        for index, trace in enumerate(self.traces.values(), start=1):
        
            print(f"Final transition for trace {index}: {trace[len(trace)-1]}")

            self.final_transitions.add(trace[len(trace)-1])
        return self.final_transitions
    


    def getFootprint_regular_alpha_miner(self) -> np.ndarray: 
        
        #Step 1: remove duplicate traces
        no_duplicate_traces = SortedSet()
        for trace in self.traces.values():
            no_duplicate_traces.add("".join(trace))
            print(no_duplicate_traces)

        print("all traces: ", no_duplicate_traces)
       
        #Extract relations between each transitions
        # generate Footprint
        # Step 1: get all direct following relations, create a sorted set for those
        print(self.transitions)
       
        successors= SortedSet()
        for traces in no_duplicate_traces:
            print(no_duplicate_traces)
            print(traces)
            # generate the successors by checking every trace
            for i in range(len(traces) - 1):
                
                successors.add(traces[i:i+2])
                print(successors)
            # self.relations[trace] = SortedDict()
        
        print(successors)
        
        # Step 2 generate the potential parallelism set aswell as the sequential task relation set
        potential_parallelism = SortedSet()
        
        for pairs in successors:
            print(pairs)
            
            for i in range(len(pairs)-1):

                
                reverse_pair = pairs[::-1]
               
             
                if reverse_pair in successors:
                    print(reverse_pair)
                    
                    potential_parallelism.add(pairs)
                    

        print(potential_parallelism)
        print("potential parallelisms: ", potential_parallelism)
        sequential_task_relations = successors - potential_parallelism
        print("sequential task relations: ", sequential_task_relations)
        directly_before_relations= SortedSet(value[::-1] for value in sequential_task_relations) 
        print("directly_before_relations: " , directly_before_relations )
        # create "not followed_relations" which is a 2 element cartesian product of all self.transitions minus the rest
        cartesian = SortedSet([char1 + char2 for char1 in self.transitions for char2 in self.transitions])
        print(cartesian)
        not_followed = cartesian - potential_parallelism - directly_before_relations - sequential_task_relations
        print("not followed_relations: " ,not_followed)

        footprint_matrix = np.array([])
        for i in cartesian:
            if i in not_followed:
                
                footprint_matrix =np.append(footprint_matrix, '#')
                print(footprint_matrix)
            elif i in sequential_task_relations:
              
                footprint_matrix =np.append(footprint_matrix, '->')
            elif i in directly_before_relations:
              
                footprint_matrix =np.append(footprint_matrix, '<-')
            elif i in potential_parallelism:
                
                footprint_matrix =np.append(footprint_matrix, '||')

        
        print(footprint_matrix.reshape((len(self.transitions), -1)))
           

        self.footprint_matrix_reshaped = footprint_matrix.reshape((len(self.transitions), -1))
       
        return self.footprint_matrix_reshaped
    

    def getFootprint(self) -> np.ndarray: 
        #Step 1: remove duplicate traces
        traces_without_duplicates = SortedSet()
        for trace in self.traces.values():
            traces_without_duplicates.add("".join(trace))
      

        print("all traces: ", traces_without_duplicates)

        #Step 2: Due to loop completeness, (definition 3.3: Ordering relations capturing length 2 loops) we need to examine all possible 
        # firing sequences that have the form: aba with  t_i-1 = a ;   t_i = b   ;  t_i+1 = a
         #Extract relations between each transitions
        # generate Footprint

        for transition_1 in self.transitions:
            self.relations[transition_1] = SortedDict()
            for transition_2 in self.transitions:
                two_element_transitions = transition_1+transition_2
                three_element_transition_1 = transition_2+transition_1+transition_2   
                # eg aba, we also need bab
                three_element_transition_2 = transition_1+transition_2+transition_1

                print(two_element_transitions)
                print(three_element_transition_1)
                print(three_element_transition_2)



                all_relations = None
                for trace in traces_without_duplicates:
                    
                    if all_relations == None :
                        if trace.find(two_element_transitions) >= 0:
                          
                            all_relations = "->"
                          
                        elif trace.find(two_element_transitions[::-1]) >= 0:
                           
                            all_relations = "<-"
                    else:
                        if trace.find(two_element_transitions) >= 0:
                            if all_relations == "<-": 
                                if trace.find(three_element_transition_1) <= 0 and trace.find(three_element_transition_2) <= 0:
                                  
                                    all_relations = "||"
                        elif trace.find(two_element_transitions[::-1]) >= 0:
                            if all_relations == "->":
                                if trace.find(three_element_transition_1) <= 0 and trace.find(three_element_transition_2) <= 0:
                               
                                    all_relations = "||"                      

                if all_relations == None:
                 
                    all_relations = "#"
                self.relations[transition_1][transition_2] = all_relations

        print(self.relations)
      
        return self.relations

        
         
    """ def visualize_footprint_matrix(self): #TODO
        self.dot = graphviz.Digraph()
        print(self.transitions)
        row_labels = column_labels =  [f'Activity {index+1}' for index, _ in enumerate(self.transitions)]
        # Add nodes (activities)
        print(row_labels)
        for label in row_labels:
            self.dot.node(label) 

     
        # Add edges (footprint values)
        print("current footprint matrix", self.footprint_matrix_reshaped)
        for i, row in enumerate(self.footprint_matrix_reshaped):
            for j, value in enumerate(row):
                if value != 0:
                    self.dot.edge(row_labels[i], column_labels[j], label=str(value))

        print(self.dot)
        
        return self.dot """









 #to test the log from the alpha plus miner paper
#with open("Logs/log_from_paper.csv","r") as my_file :

with open("Logs/log.csv","r") as my_file :
    traces = SortedDict()
    logdata = my_file.read()
    print(logdata)
   # print(logdata[3:-1])
    # remove first line from the csv in case of problems with the formatting
   # logdata2 = logdata[3:-1]
   # logdata = logdata2
   # import time
   # time.sleep(5)
    events =logdata.split("\n")
    for event in events:
        
        print(event)
        case_id,activity = event.split(',')
        if case_id not in traces:
            traces[case_id] = []

        traces[case_id].append(activity)
    
    
    print(traces) 
    
   
    



alphaminerplusobject = AlphaMinerplus(traces)

print(alphaminerplusobject.getInitialTransitions())  
print(alphaminerplusobject.getFinalTransitions())  
print(alphaminerplusobject.getTransitions()) 

print(alphaminerplusobject.getFootprint())





#print(alphaminerplusobject.visualize_footprint_matrix())




# Example labels for rows and columns
#row_labels = [f'Activity {i+1}' for i in self.transitions]
#column_labels = ['Activity 1', 'Activity 2', 'Activity 3']
# Visualize the footprint matrix
#graph = alphaminerplusobject.visualize_footprint_matrix()
#graph.render('footprint_matrix', format='png', cleanup=True)  # Render the graph to a file
#visualizer.apply(alphaminerplusobject.getFootprint())