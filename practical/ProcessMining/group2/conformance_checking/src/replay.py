from pm4py.algo.simulation.playout.petri_net import algorithm
import imported_pm4py.extensive as extensive
from pm4py.objects.petri_net.semantics import is_enabled, execute
from collections import deque
from pm4py.objects.petri_net.obj import PetriNet
from pm4py.objects.log.obj import Trace, Event, EventLog
from pm4py.algo.simulation.playout.petri_net import algorithm


def get_traces_with_replay(net, start, end, variant):
    print("Replaying Petri Net with TQDM")
    playout = extensive.apply(net, start, end)
    return playout
