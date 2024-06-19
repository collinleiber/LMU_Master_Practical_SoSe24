from practical.ProcessMining.group2.conformance_checking import Conformance_checking


from sortedcontainers import SortedDict, SortedSet



def test_cf():
    traces = SortedDict({'1': ['a', 'b', 'c', 'd'], '10': ['a', 'b', 'c', 'd'], '11': ['d', 'a', 'b'], '12': ['d', 'a', 'b'], '13': ['d', 'a', 'b'], '14': ['d', 'a', 'b'], '15': ['d', 'a', 'b'], '16': ['d', 'a', 'b'], '17': ['d', 'a', 'b'], '18': ['d', 'a', 'b'], '19': ['d', 'a', 'b'], '2': ['a', 'b', 'c', 'd'], '20': ['d', 'a', 'b'], '21': ['a', 'd', 
    'c'], '22': ['a', 'd', 'c'], '23': ['a', 'd', 'c'], '24': ['a', 'd', 'c'], '25': ['a', 'd', 'c'], '26': ['a', 'd', 'c'], '27': ['a', 'd', 'c'], '28': ['a', 'd', 'c'], '29': ['a', 'd', 'c'], '3': ['a', 'b', 'c', 'd'], '30': ['a', 'd', 'c'], '31': ['b', 'c', 'd'], '32': ['b', 'c', 'd'], '33': ['b', 'c', 'd'], '34': ['b', 'c', 'd'], '35': ['b', 'c', 'd'], '4': ['a', 'b', 'c', 'd'], '5': ['a', 'b', 'c', 'd'], '6': ['a', 'b', 'c', 'd'], '7': ['a', 'b', 'c', 'd'], '8': ['a', 'b', 'c', 'd'], '9': ['a', 'b', 'c', 'd']})     
   

    cf_object = Conformance_checking(traces)


    cf_object.get_transitions()

    


    cf_object.get_footprint_regular_alpha_miner()
    
    assert cf_object.relations == SortedDict({'a': SortedDict({'a': '#', 'b': '->', 'c': '#', 'd': '||'}), 'b': SortedDict({'a': '<-', 'b': '#', 'c': '->', 'd': '#'}), 'c': SortedDict({'a': '#', 'b': '<-', 'c': '#', 'd': '->'}), 'd': SortedDict({'a': '||', 'b': '#', 'c': '<-', 'd': '#'})})



        
    


    
    cf_object.visualize_sorted_dict(cf_object.relations)