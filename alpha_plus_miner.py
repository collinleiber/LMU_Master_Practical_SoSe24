
from enum import Enum
from sortedcontainers import SortedSet, SortedDict
import numpy as np
import time
import pm4py
from pm4py.visualization.footprints import visualizer
#import graphviz
#from pm4py.objects.log.importer.xes import factory as xes_importer

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
        # set of pairs 
        self.pairs = []

        # set of maximal pairs
        self.maximal_pairs = []

        # set of places between maximal pairs 
        self.places = []

        self.length_one_loops = None

        # This corresponds to step 3 of the alpha plus algorithm:  3) T' := T \ L1L 
        self.log_without_length_one_loops = None   

        self.F_L1L = None

        self.W_minusL1L = SortedDict()

        
        
       # self.footprint_matrix_reshaped #= self.getFootprint()

    def getTransitions(self) -> SortedSet: 
        #returns all transitions for the current petri net
        for trace in self.traces.values():
          #  print("current trace: ", trace)
            for activity in trace:
                self.transitions.add(activity)
            #    print("current activity: " , activity)

       # print("getTransition returns: ", self.transitions)
        return self.transitions
    

    def getInitialTransitions(self) -> SortedSet:
        
        #For each trace get the first activity and add it to the set of initial transitions
        for index, trace in enumerate(self.traces.values(), start=1):
          #  print(f"Initial transition for trace {index}: {trace[0]}")
           
            self.initial_transitions.add(trace[0])
        return self.initial_transitions

    def getFinalTransitions(self) -> SortedSet:
        #For each trace get the last activity and add it to the set of final transitions
        for index, trace in enumerate(self.traces.values(), start=1):
        
          #  print(f"Final transition for trace {index}: {trace[len(trace)-1]}")

            self.final_transitions.add(trace[len(trace)-1])
        return self.final_transitions
 
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

          #      print(two_element_transitions)
           #     print(three_element_transition_1)
            #    print(three_element_transition_2)



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

        #print(self.relations)
      
        return self.relations

        
      






    def getPairs(self):
        # generate pairs of activities, same procedure as the regular alpha miner 
        #There must not be a relation between the activities
        #additinally the activities in the set have to be direcly successed by each other 

      # Find pairs (A, B) of sets of activities such that every element a ∈ A and every element b ∈ B are
      #causally related (i.e. a →L b), all elements in A are independent (a1#a2), and all elements in B
      #are independent (b1#Lb2) as well


        choice_pairs = []

    
        causality_pairs = []
        
        #Extract all possible pairs of activities with the causality relation
        for activity1 ,relations1 in self.relations.items():
          #  print("activity1: ", activity1)
         #   print("relation1: ",  relations1)
           
            for activity2 , relation in relations1.items():
              #  print("activity2: ", activity2)
              #  print("relation: ",  relation)
            
                if relation == "->" :
                    causality_pairs.append((activity1,activity2))
                if relation == "#":
                    if activity1 == activity2:
                        choice_pairs.append((activity1,))
                    else:
                        choice_pairs.append((activity1,activity2))

      #  print("causality_pairs: ", causality_pairs)
        pairs = causality_pairs
     #   print("choice_pairs: ", choice_pairs)
        #causality_pairs:  [('a', 'b'), ('a', 'c'), ('a', 'e'), ('b', 'd'), ('c', 'd'), ('e', 'd')]
#       choice_pairs:  [('a',), ('a', 'd'), ('b',), ('b', 'e'), ('c',), ('c', 'e'), ('d', 'a'), ('d',), ('e', 'b'), ('e', 'c'), ('e',)]



        # log from the paper: 
   #     causality_pairs:  [('a', 'b'), ('a', 'c'), ('b', 'd'), ('c', 'd'), ('e', 'f')]
#       choice_pairs:  [('a',), ('a', 'd'), ('a', 'e'), ('a', 'f'), ('b',), ('b', 'e'), ('b', 'f'), ('c',), ('c', 'e'), ('c', 'f'), ('d', 'a'), ('d',), 
      #  ('d', 'e'), ('d', 'f'), ('e', 'a'), ('e', 'b'), ('e', 'c'), ('e', 'd'), ('e',), ('f', 'a'), ('f', 'b'), ('f', 'c'), ('f', 'd'), ('f',)]
     
        # We need to check the combinations too, while checking if the activities are independent in their respective sets


       
        i = 0
        j = len(choice_pairs)

        while i < j :
            current_choice_pair = choice_pairs[i]
            for pair in choice_pairs:
                union = True
            #    print(SortedSet(current_choice_pair))
             #   print(SortedSet(pair))
             #   print(len(SortedSet(current_choice_pair).intersection(SortedSet(pair))))
           
                if len(SortedSet(current_choice_pair).intersection(SortedSet(pair))) != 0:
                    for e1 in pair:
                        if union == False:
                            break
                        for e2 in current_choice_pair:
                            if self.relations[e1][e2] != "#":  #check if the activities are independent
                                union = False
                                break
                    if union :  # if not independent, create a new pair which is the union of the 2 activities eg  [a,d]  +  [a, f] -> [a, d, f]
                        
                        new_pair = SortedSet(current_choice_pair) | SortedSet(pair) 
                   #     print("potential new constructed pair: ", new_pair)
                      
                        if tuple(new_pair) not in choice_pairs:
                      #      print("new_pair: ", new_pair)
                        
                            choice_pairs.append(tuple(new_pair))
                            j = j + 1
                            
                
            i = i + 1

        print("choice_pairs: ", choice_pairs)
        #choice_pairs:  [('a',), ('a', 'd'), ('b',), ('b', 'e'), ('c',), ('c', 'e'), ('d', 'a'), ('d',), ('e', 'b'), ('e', 'c'), ('e',)]


       # log from the paper
   #    choice_pairs:  [('a',), ('a', 'd'), ('a', 'e'), ('a', 'f'), ('b',), ('b', 'e'), ('b', 'f'), ('c',), ('c', 'e'), ('c', 'f'), ('d', 'a'), ('d',), 
   #('d', 'e'), ('d', 'f'), ('e', 'a'), ('e', 'b'), ('e', 'c'), ('e', 'd'), ('e',), ('f', 'a'), ('f', 'b'), ('f', 'c'), ('f', 'd'), ('f',),
   ##  ('a', 'd', 'e'), ('a', 'd', 'f')]
   # ade and adf have been successfully added
      
         # First we did the check for #, which is why we get ('a', 'd', 'e'), ('a', 'd', 'f')
         # now follows the check if they have "->"


        # Union 
        for pair_choices1 in choice_pairs:
            for pair_choices2 in choice_pairs:
                relation_between_pair = None
                makePair = True
             #   print("pair 1 ",pair_choices1)
             #   print("pair 2 ",pair_choices2)
             
                intersection = SortedSet(pair_choices1).intersection(pair_choices2)
             #   print("intersection: ", intersection)
                pair_choices2 = SortedSet(pair_choices2)
              #  print("pair_choices: ", pair_choices2)
              
                if len(intersection) != 0 :
                    # remove intersection terms in the second pair
                    for term in intersection:
                    #    print("term to discard:" , term)
                     
                        pair_choices2.discard(term)
                    
                if(len(pair_choices2) == 0):
                    continue
                pair_choices2= tuple(pair_choices2)
              #  print("pair_choices2 with discarded items :",pair_choices2)

                for activity1 in pair_choices1:
                #    print(activity1)
                    if makePair == False:
                        break
                    for activity2 in pair_choices2:
                   #     print(activity2)
                        relation = self.relations[activity1][activity2]
                        if relation_between_pair != None and relation_between_pair != relation:
                            makePair = False
                            break
                        else:
                            relation_between_pair = relation
                        if relation != "->":
                            makePair = False
                            break
                if makePair == True:
                 #   print("makepair reached")
                  #  print(pair_choices1)
                  #  print(pair_choices2)
                    
                    if relation_between_pair == "->":
                        new_pair = (pair_choices1,pair_choices2)
                    else:
                        new_pair = (pair_choices2,pair_choices1)
                    pairs.append(new_pair)
      
      
        print("pairs from getPairs: ", pairs)
     
        self.pairs = pairs




    def get_maximal_pairs(self):
          #  Set of paired activities that are maximal
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
             #   print("pair1 :",pair1)
            #    print("flat_pair1 :",flat_pair1)
            
                pos2 = 0
                for pair2 in self.pairs:
                    if pos1 != pos2:
                        flat_pair2 = []
                        for pair_element in pair2:
                            for e in pair_element:
                                flat_pair2.append(e)
                    #    print("pair2 :",pair2)
                     #   print("flat_pair2 :",flat_pair2)
                        # flat the pair 1

                        # flat the pair 2
                        # check if pair1 issubset of pair 2 or pair 2 is subset of 1
                        if SortedSet(flat_pair1).issubset(flat_pair2) and SortedSet(flat_pair1)!= SortedSet(flat_pair2):
                      #      print("is a subset")
                            append_flag = False
                    pos2 = pos2 + 1

                if append_flag == True:

             #       print("append")
                    if SortedSet(flat_pair1) not in pair_appended:
                        maximal_pairs.append(pair1)
                        pair_appended.append(SortedSet(flat_pair1))
                pos1 = pos1 + 1
        #    print("maximal_pairs:" , maximal_pairs)
          
            self.maximal_pairs = maximal_pairs








    def get_length_one_loops(self) -> SortedSet:
        #extract length one loop 
        self.length_one_loops = SortedSet()
        self.getTransitions()
        #compute footprint matrix and extract all transitions that have a causality relation with itself, eg: aa, bb etc.
       
        self.getFootprint()
       
        for transition in self.transitions:
            if self.relations[transition][transition] == "->":
                self.length_one_loops.add(transition)

     #   print("Length 1 loop: ", self.length_one_loops)
        return self.length_one_loops



    
    def remove_length_one_loops(self) ->SortedSet:
        # This function corresponds to step 3 of the alpha plus algorithm:  3) T' := T \ L1L 
        self.log_without_length_one_loops = self.transitions.difference(self.length_one_loops)
     #   print(self.transitions)
      #  print(self.log_without_length_one_loops)
        
        return self.log_without_length_one_loops




    def get_FL1L(self):

        # (You need T' for this )
       # Step 5 of the algorithm:
       # For each t ∈ L1L do:
        #(a) A = {a ∈ T'  | a >W t}
        #(b) B = {b ∈ T'  | t >W a}
        #(c) FL1L := FL1L ∪ {(t, p(A\B,B\A)),(p(A\B,B\A),t)}


        self.F_L1L = SortedSet()
        place_counter = 1
        self.getTransitions()
        for transition1 in self.length_one_loops:

            A = SortedSet()
            B = SortedSet()
            for transition2 in self.log_without_length_one_loops:
                if self.relations[transition2][transition1] == "->":
                  #  print("for transition ",transition1," : ",transition2)
                    A.add(transition2)
                if  self.relations[transition1][transition2] == "->":
                  #  print("for transition ",transition1," : ",transition2)
                    B.add(transition2)
           

            print(len(A) == len(B))
            
            place = 'p'+str(place_counter)
            for _ in A.difference(B):
                # Add input places
                transition_place = (transition1,place)
                self.F_L1L.add(transition_place) 
            for _ in B.difference(A):
                #Add output place
                transition_place = (place,transition1)
                self.F_L1L.add(transition_place)
          

            place_counter += 1
      #  print(self.F_L1L)  
      





    
    # We want the difference between 2 lists in generate_W_minus_L1L
    def diff(self,first, second) -> list:
        second = set(second)
        return [item for item in first if item not in second]
    

#    example functionality: trace ['a', 'c', 'c', 'b', 'b', 'd', 'e']
#    length one loops SortedSet(['b', 'c'])
#    trace without length one loops ['a', 'd', 'e']
    def generate_W_minus_L1L(self):
        #W_minusL1L
        length_one_loops = self.length_one_loops
      #  print('length one loops',length_one_loops)
        for trace_key,trace in self.traces.items():
             trace_pr = trace
            # print('trace',trace_pr)
            
            
             trace_pr = self.diff(trace_pr,length_one_loops)
            # print('trace without length one loops',trace_pr)
             self.W_minusL1L[trace_key] = trace_pr
       
        #print('W_minusL1L ',self.W_minusL1L)


    def add_places(self):
        

        # connect the initial transition with the first place, after the for loop we connect the last transition to the last place 
        place_counter = 0
        self.places.append(("Place_"+str(place_counter),self.initial_transitions))
        place_counter = 1  
        for pair in self.maximal_pairs:
            self.places.append((pair[0],"Place_"+str(place_counter),pair[1]))
            place_counter =  place_counter+1
        self.places.append((self.final_transitions,"Place_"+str(place_counter)))
        print("input for the visualisation: ", self.places)







    def run_alphaMiner_plus(self):
                
        alphaminerplusobject = AlphaMinerplus(self.W_minusL1L)
      
        alphaminerplusobject.getInitialTransitions() 
        alphaminerplusobject.getFinalTransitions()
        alphaminerplusobject.getTransitions()
        alphaminerplusobject.getFootprint()


            

        alphaminerplusobject.getPairs()
        alphaminerplusobject.get_maximal_pairs()

        alphaminerplusobject.add_places()


        #TODO add visualization here



""" 

events_xes = pm4py.read_xes("Logs/pdc2023_000000.xes")

# Extract case IDs and activities
case_ids = events_xes['case:concept:name']
activities = events_xes['concept:name']
# test without unique()
# Print case IDs and activities
print("Case IDs:")
print(case_ids)
print("\nActivities:")
print(activities)


traces = SortedDict()
# Group by case IDs and iterate through each group
for case_id, group in events_xes.groupby('case:concept:name'):
    activities = group['concept:name'].unique()
    
    

    # Remove "trace" and the space
    print("case_id before: ", case_id)
    cleaned_string = case_id.replace("trace", "").replace(" ", "")
    print("case_id ", cleaned_string)
 
    print(activities)
   
    activities_with_commas = ", ".join(activities.tolist())
    activities_test = ", ".join(activities)
   
    
    act = activities_test.split(", ")
    
    if cleaned_string not in traces:
            traces[cleaned_string] = [] 
            
   

            traces[cleaned_string].extend(act)
   


 """





#with open("Logs/log_from_paper.csv","r") as my_file :
#with open("Logs/simple_log.csv","r") as my_file :

with open("Logs/log_2loop.csv","r") as my_file :

#with open("Logs/test_alphaplus2.csv","r") as my_file :
    traces = SortedDict()
    logdata = my_file.read()
    print("logdata: \n", logdata)
  
    # remove first line from the csv in case of problems with the formatting
   # logdata2 = logdata[0:-1]   #Logs/simple_log.csv
   # logdata2 = logdata[3:-1]   #Logs/log_from_paper.csv

    logdata2 = logdata  ##Logs/log_2loop.csv
    logdata = logdata2
   
    events =logdata.split("\n")
    for event in events:
        
        print(event)
        case_id,activity = event.split(',')
        if case_id not in traces:
            traces[case_id] = []

        traces[case_id].append(activity)
       
        
    print("traces", traces) 
   
    
    




#alpha plus miner doesnt get regular traces as input, rather it gets self.W_minusL1L, which is step 8 of the algorithm
# α(W_−L1L)
alphaminerplusobject = AlphaMinerplus(traces)



print("Length One Loops ",alphaminerplusobject.get_length_one_loops())
print("T' ",alphaminerplusobject.remove_length_one_loops())
print("getF_L1L" ,alphaminerplusobject.get_FL1L())
print("Value for W_minus_L1L" ,alphaminerplusobject.generate_W_minus_L1L())

alphaminerplusobject.run_alphaMiner_plus()







