from pm4py.objects.petri_net.obj import PetriNet, Marking
from practical.ProcessMining.group1.shared.visualizer import Visualizer
import graphviz


class TestTokenReplay:
    def test_token_replay(self):
        pass

    def test_can_fire(self):
        pass

    def test_fire(self):
        pass

    def test_handle_tau(self):
        pass

    def test_handle_unconformity(self):
        pass

    def test_calculate_remaining_tokens(self):
        pass

    def test_get_unconformity_tokens(self):
        pass

    def test_token_replay_getters(self):
        pass

    def test_get_dimension_value(self):
        pass

    def test_calculate_fitness(self):
        pass

    def test_calculate_pm4py_dimensions(self):
        pass

    def test_shuffle_activities(self):
        pass

    def test_visualize_replay_result(self, tmp_path):
        visualizer = Visualizer()

        net = PetriNet("net1")
        p_im = PetriNet.Place("p_im")
        net.places.add(p_im)
        p1 = PetriNet.Place("p1")
        net.places.add(p1)
        p2 = PetriNet.Place("p2")
        net.places.add(p2)
        p_fm = PetriNet.Place("p_fm")
        net.places.add(p_fm)

        t1 = PetriNet.Transition("t1", "t1")
        net.transitions.add(t1)
        t2 = PetriNet.Transition("t2", "t2")
        net.transitions.add(t1)
        t3 = PetriNet.Transition("t3", "t3")
        net.transitions.add(t1)

        net.arcs.add(PetriNet.Arc(p_im, t1))
        net.arcs.add(PetriNet.Arc(t1, p1))
        net.arcs.add(PetriNet.Arc(p1, t2))
        net.arcs.add(PetriNet.Arc(t2, p2))
        net.arcs.add(PetriNet.Arc(p2, t3))
        net.arcs.add(PetriNet.Arc(t3, p_fm))

        initial_marking = Marking({p_im: 1})
        final_marking = Marking({p_fm: 0})

        tokens = {"missing": {p_im: 1, p2: 2, p_fm: 2}, "remaining": {p_im: 1, p1: 2, p_fm: 3}}

        graph = visualizer.build_petri_net(net, initial_marking, final_marking, tokens)
        print(graph.source)
        # Check if the graph is created
        assert isinstance(graph, graphviz.Digraph)
        # Check if the label is adjusted
        assert 'label=<<FONT POINT-SIZE="28">&#9679;</FONT><BR></BR><B>+1 -1</B>>' in graph.body[4]
        assert 'label=<<B>+3<BR></BR>-2</B>>' in graph.body[3]
        assert 'label=<<B>-2</B>>' in graph.body[2]
        assert 'label=<<B>+2</B>>' in graph.body[1]
        # Check if the fillcolor is adjusted
        for i in range(1, 5):
            assert 'fillcolor=transparent' not in graph.body[i]