import pytest
from pm4py.objects.petri_net.obj import PetriNet, Marking
from practical.ProcessMining.group1.shared.visualizer import Visualizer
import graphviz

from practical.ProcessMining.group1.task4.tokenreplay import TokenReplay


class TestTokenReplay:
    @pytest.fixture
    def token_replay(self):
        net = PetriNet("net1")

        p_im = PetriNet.Place("p_im")
        p1 = PetriNet.Place("p1")
        p2 = PetriNet.Place("p2")
        p_fm = PetriNet.Place("p_fm")
        net.places.add(p_im)
        net.places.add(p1)
        net.places.add(p2)
        net.places.add(p_fm)

        t1 = PetriNet.Transition("t1", "t1")
        t2 = PetriNet.Transition("t2", "t2")
        t3 = PetriNet.Transition("t3", "t3")
        net.transitions.add(t1)
        net.transitions.add(t2)
        net.transitions.add(t3)

        arc1 = PetriNet.Arc(p_im, t1)
        arc2 = PetriNet.Arc(t1, p1)
        arc3 = PetriNet.Arc(p1, t2)
        arc4 = PetriNet.Arc(t2, p2)
        arc5 = PetriNet.Arc(p2, t3)
        arc6 = PetriNet.Arc(t3, p_fm)
        net.arcs.add(arc1)
        net.arcs.add(arc2)
        net.arcs.add(arc3)
        net.arcs.add(arc4)
        net.arcs.add(arc5)
        net.arcs.add(arc6)

        p_im.out_arcs.add(arc1)
        t1.in_arcs.add(arc1)
        t1.out_arcs.add(arc2)
        p1.in_arcs.add(arc2)
        p1.out_arcs.add(arc3)
        t2.in_arcs.add(arc3)
        t2.out_arcs.add(arc4)
        p2.in_arcs.add(arc4)

        initial_marking = Marking({p_im: 1})
        final_marking = Marking({p_fm: 0})

        log = []

        tr = TokenReplay(log, net, initial_marking, final_marking, "testFixture")
        tr.missing_tokens = {p_im: 1, p2: 2, p_fm: 2}
        tr.remaining_tokens = {p_im: 1, p1: 2, p_fm: 3}

        return tr

    def test_token_replay(self):
        pass

    def test_can_fire(self, token_replay):
        assert token_replay._can_fire('t1')
        assert not token_replay._can_fire('t2')
        assert not token_replay._can_fire('unknown')

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

    def test_visualize_replay_result(self, tmp_path, token_replay):
        visualizer = Visualizer()

        graph = visualizer.build_petri_net(token_replay.net, token_replay.initial_marking, token_replay.final_marking,
                                           token_replay.get_unconformity_tokens())
        print(graph.source)
        # Check if the graph is created
        assert isinstance(graph, graphviz.Digraph)
        # Check if the label is adjusted
        assert 'label=<<FONT POINT-SIZE="28">&#9679;</FONT><BR></BR><B>+1 -1</B>>' in graph.body[6]
        assert 'label=<<B>+3<BR></BR>-2</B>>' in graph.body[5]
        assert 'label=<<B>-2</B>>' in graph.body[4]
        assert 'label=<<B>+2</B>>' in graph.body[3]
        # Check if the fillcolor is adjusted
        for i in range(3, 6):
            assert 'fillcolor=transparent' not in graph.body[i]