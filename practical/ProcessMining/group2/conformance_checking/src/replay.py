from pm4py.objects.petri_net.semantics import is_enabled, execute
from collections import deque
from pm4py.analysis import get_enabled_transitions
from pm4py.objects.log.obj import Trace, Event, EventLog


def get_traces_with_replay(net, start, end, max_depth=100):
    print("Replaying Petri Net with max_depth={}".format(max_depth))
    log = EventLog()
    queue = deque([(start, [])])
    trace_num = 1

    while queue:
        current_place, current_trace = queue.popleft()

        if current_place == end:
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

        enabled = get_enabled_transitions(net, current_place)

        # Breadth-first search
        for transition in enabled:
            new_place = execute(transition, net, current_place)
            queue.append((new_place, current_trace + [transition.name]))

    return log
