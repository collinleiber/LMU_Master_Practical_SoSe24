from alpha_plus_miner import AlphaMinerplus as amp
from sortedcontainers import SortedSet, SortedDict

with open("Logs/log.csv","r") as my_file :
    traces = SortedDict()
    logdata = my_file.read()
  
    events =logdata.split("\n")
    for event in events:
        case_id,activity = event.split(',')
        if case_id not in traces:
            traces[case_id] = []

        traces[case_id].append(activity)
       
#alpha plus miner doesnt get regular traces as input, rather it gets self.W_minusL1L, which is step 8 of the algorithm
# α(W_−L1L)
alpha_miner = amp(traces)

def test_initiate_alpha_miner():
    # Test if the AlphaMinerplus class can be initiated
    alpha_miner = amp(traces)
    assert alpha_miner is not None
    assert alpha_miner.traces == traces
    
def test_get_transitions():
    # Test if the transitions are extracted correctly
    alpha_miner = amp(traces)
    # TODO - Get correct transitions for the given log

    # assert alpha_miner.getInitialTransitions() == SortedSet(['a', 'b'])
    # assert alpha_miner.getTransitions() == SortedSet(['a', 'b', 'c', 'd'])
    # assert alpha_miner.getFinalTransitions() == SortedSet(['c', 'd'])

def test_get_footprint():
    # Test if the footprint matrix is extracted correctly
    alpha_miner = amp(traces)
    footprint = alpha_miner.getFootprint()
    # TODO - Implement after refactoring the code
    # assert footprint == None

def test_get_pairs():
    # Test if the pairs are extracted correctly
    alpha_miner = amp(traces)
    # TODO - Implement after refactoring the code
    # assert alpha_miner.pairs == []

def test_visualization():
    # Test if the visualization function is working correctly
    alpha_miner = amp(traces)
    alpha_miner.visualization()
    # TODO - Implement after refactoring the code
    # assert alpha_miner.visualize() == None

def test_run_alphaMiner_plus():
    # Preprocessing
    print("Length One Loops ",alpha_miner.get_length_one_loops())
    print("T' ",alpha_miner.remove_length_one_loops())
    print("getF_L1L" ,alpha_miner.get_FL1L())
    print("Value for W_minus_L1L" ,alpha_miner.generate_W_minus_L1L())

    alphaminerplusobject = amp(alpha_miner.W_minusL1L)
      
    alphaminerplusobject.getInitialTransitions() 
    alphaminerplusobject.getFinalTransitions()
    alphaminerplusobject.getTransitions()
    alphaminerplusobject.getFootprint()

    alphaminerplusobject.getPairs()
    alphaminerplusobject.get_maximal_pairs()

    alphaminerplusobject.add_places()

    alphaminerplusobject.visualization()