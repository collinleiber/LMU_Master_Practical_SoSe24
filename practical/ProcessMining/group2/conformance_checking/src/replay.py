import imported_pm4py.extensive as extensive
import imported_pm4py.basic_playout as basic

from pm4py.algo.simulation.playout.petri_net import variants


def get_traces_with_replay(net, start, end, variant=variants.extensive):
    """
    Replay a Petri net to obtain traces using the specified replay variant.

    This function replays the given Petri net model and retrieves the resulting traces
    based on the selected replay variant.

    Parameters:
    ----------
    net : PetriNet
        The Petri net model to be replayed.
    start : Marking
        The initial marking of the Petri net.
    end : Marking
        The final marking of the Petri net.
    variant : variants, optional
        The variant of the replay algorithm to use. Default is variants.extensive.

    Returns:
    -------
    list
        The replayed traces obtained from the Petri net.
    """
    print("Replaying Petri Net with TQDM")
    if variant == variants.basic_playout:
        playout = basic.apply(net, start, end)
    else:
        playout = extensive.apply(net, start, end)
    return playout
