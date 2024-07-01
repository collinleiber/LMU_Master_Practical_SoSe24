import imported_pm4py.extensive as extensive
import imported_pm4py.basic_playout as basic

from pm4py.algo.simulation.playout.petri_net import variants


def get_traces_with_replay(net, start, end, variant=variants.extensive):
    print("Replaying Petri Net with TQDM")
    if variant == variants.basic_playout:
        playout = basic.apply(net, start, end)
    else:
        playout = extensive.apply(net, start, end)
    return playout
