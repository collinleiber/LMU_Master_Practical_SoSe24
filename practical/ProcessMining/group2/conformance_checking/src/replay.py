from pm4py.objects.petri_net.semantics import is_enabled, execute
from collections import deque
from pm4py.analysis import get_enabled_transitions


def get_traces_with_replay(net, start, end, max_depth=100):
    traces = []
    queue = deque([(start, [])])

    while queue:
        current_place, current_trace = queue.popleft()

        print("Now Checking: ", current_place, "Current Trace: ", current_trace)

        if current_place == end:
            traces.append(current_trace)
            continue

        if len(current_trace) >= max_depth:
            continue

        enabled = get_enabled_transitions(net, current_place)

        # Breadth-first search
        for transition in enabled:
            new_place = execute(transition, net, current_place)
            queue.append((new_place, current_trace + [transition.name]))

    return traces
