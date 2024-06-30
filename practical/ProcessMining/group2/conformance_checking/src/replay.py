from pm4py.algo.simulation.playout.petri_net import algorithm
import imported_pm4py.extensive as extensive
from pm4py.objects.petri_net.semantics import is_enabled, execute
from collections import deque
from pm4py.objects.petri_net.obj import PetriNet
from pm4py.objects.log.obj import Trace, Event, EventLog
from pm4py.algo.simulation.playout.petri_net import algorithm


def get_traces_with_replay(net, start, end, variant):
    print("Replaying Petri Net with TQDM")
    playout = extensive.apply(net, start,end)
    return playout


def get_directly_connected_transitions(place):
    directly_connected_transitions = set()
    
    if isinstance(place, PetriNet.Place):
        for arc in place.out_arcs:
            transition = arc.target
            directly_connected_transitions.add(transition)
    
    return list(directly_connected_transitions)

def get_traces_with_replay_own_implementation(net, start, end, max_depth=100):
    print("Replaying Petri Net with max_depth={}".format(max_depth))
    log = EventLog()
    queue = deque([(start, [])])
    trace_num = 1

    while queue:
        current_marking, current_trace = queue.popleft()
        if current_marking == end:
            trace = Trace()
            trace.attributes['concept:name'] = str(trace_num)
            for activity in current_trace:
                event = Event({'concept:name': activity})
                trace.append(event)
            log.append(trace)
            trace_num += 1
            continue

        if len(current_trace) >= max_depth:
            continue

        for place in current_marking.keys():
            enabled_transitions = get_directly_connected_transitions(place)
            for transition in enabled_transitions:
                if is_enabled(transition, net, current_marking):
                    new_marking = current_marking.copy()
                    execute(transition, net, new_marking)
                    
                    if transition.label is None:
                        queue.append((new_marking, current_trace))
                    else:
                        queue.append((new_marking, current_trace + [transition.label]))

    return log

def get_traces_with_replay_pm4py(net, start, end, variant):
    print("Replaying Petri Net")
    playout = algorithm.apply(net, start,end,variant=variant)
    return playout
