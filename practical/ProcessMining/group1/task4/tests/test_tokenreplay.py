import graphviz
import pm4py
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.conversion.log import converter as log_converter
import pytest

from practical.ProcessMining.group1.shared import utils
from practical.ProcessMining.group1.shared.visualizer import Visualizer
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

    def test_token_replay(self, token_replay):
        # Run attempt with initially empty log
        assert not token_replay.run()

        log = [("p_im", "p1", "p2"), ("p_im", "tau", "p_fm"), ("p2",)]
        log = utils.event_log_to_pm4py_dataframe(log)
        event_log = log_converter.to_event_log.apply(log)

        net, initial, final = pm4py.discover_petri_net_alpha(event_log)

        token_replay = TokenReplay(event_log, net, initial, final, "alphaTest")

        token_replay.run(event_log)

        assert sum(x for x in dict(token_replay.missing_tokens).values()) == 4
        assert sum(y for y in dict(token_replay.remaining_tokens).values()) == 1

    def test_can_fire(self, token_replay):
        assert token_replay._can_fire('t1')
        assert not token_replay._can_fire('t2')
        assert not token_replay._can_fire('unknown')

    def test_fire(self, token_replay):
        initial_marking = token_replay.marking.copy()
        token_replay._fire('t1')
        # Get the 't1' transition from the net
        t1 = next((t for t in token_replay.net.transitions if t.label == 't1'), None)

        # Check if the marking of the input place of 't1' has been decremented by 1
        for arc in t1.in_arcs:
            assert token_replay.marking[arc.source] == initial_marking[arc.source] - 1

        # Check if the marking of the output place of 't1' has been incremented by 1
        for arc in t1.out_arcs:
            assert token_replay.marking[arc.target] == initial_marking[arc.target] + 1

        assert not token_replay._fire('no_transition')

    def test_handle_tau(self, token_replay):
        # Create a trace with a tau event followed by a 't1' event
        trace = ['tau', 't1']
        pointer = 0
        initial_produced = token_replay.produced_buffer
        initial_consumed = token_replay.consumed_buffer

        token_replay._handle_tau(trace, pointer)

        # Get the 't1' transition from the net
        t1 = next((t for t in token_replay.net.transitions if t.label == 't1'), None)

        # Check if the marking of the input place of 't1' has been decremented by 1
        for arc in t1.in_arcs:
            assert token_replay.marking[arc.source] == 0

        # Check if the marking of the output place of 't1' has been incremented by 1
        for arc in t1.out_arcs:
            assert token_replay.marking[arc.target] == 1

        # Check if produced_buffer and consumed_buffer have been incremented by 2, for tau and following
        assert token_replay.produced_buffer == initial_produced + 2
        assert token_replay.consumed_buffer == initial_consumed + 2

        current_produced = token_replay.produced_buffer
        current_consumed = token_replay.consumed_buffer

        # Create a failing trace with a tau event followed by a non-existent event
        failing_trace = ['tau', 'non_existent_event']
        pointer = 0

        token_replay._handle_tau(failing_trace, pointer)

        assert token_replay.produced_buffer == current_produced
        assert token_replay.consumed_buffer == current_consumed

    @pytest.mark.parametrize(
        "trace",
        [
            ['t1', 'p1'],  # existing transition
            ['non_existent_event', 'p1'],  # not existing
            ['p_im', 'p_fm']
        ]
    )
    def test_handle_missing_event(self, token_replay, trace):
        # Create a trace with a non-existent event
        pointer = 0
        initial_missing_tokens = token_replay.missing_tokens.copy()

        token_replay._handle_missing_event(trace[pointer])

        # Get the 'non_existent_event' transition from the net
        transition = next((t for t in token_replay.net.transitions if t.label == trace[pointer]), None)

        # If the transition exists, check if missing_tokens have been incremented for each input place of the transition
        if transition:
            for arc in transition.in_arcs:
                if token_replay.marking[arc.source] < 1:
                    assert token_replay.missing_tokens[arc.source] == initial_missing_tokens[arc.source] + 1
                else:
                    assert token_replay.missing_tokens[arc.source] == initial_missing_tokens[arc.source]
        # If the transition does not exist, check if missing_tokens have not been changed
        else:
            assert token_replay.missing_tokens == initial_missing_tokens

    def test_calculate_remaining_tokens(self, token_replay):
        remaining_tokens = token_replay._calculate_remaining_tokens()

        assert isinstance(remaining_tokens, dict)

        total_remaining = sum(val for val in remaining_tokens.values())
        assert sum(val for val in token_replay.remaining_tokens.values()) - total_remaining == 5

    def test_get_unconformity_tokens(self, token_replay):
        tokens = token_replay.get_unconformity_tokens()

        assert isinstance(tokens, dict)
        assert "missing" in tokens and "remaining" in tokens
        assert tokens["missing"] == token_replay.missing_tokens
        assert tokens["remaining"] == token_replay.remaining_tokens

    def test_token_replay_getters(self, token_replay):
        fitness = token_replay.get_fitness()
        assert isinstance(fitness, float) or isinstance(fitness, int)
        assert 1 >= fitness >= 0

        precision = token_replay.get_precision()
        assert isinstance(precision, float) or isinstance(precision, int)
        assert 1 >= precision >= 0

        simplicity = token_replay.get_simplicity()
        assert isinstance(simplicity, float) or isinstance(simplicity, int)
        assert 1 >= simplicity >= 0

        generalization = token_replay.get_generalization()
        assert isinstance(generalization, float) or isinstance(generalization, int)
        assert 1 >= generalization >= 0

        net_type = token_replay.get_discovery_type()
        assert isinstance(net_type, str)
        assert net_type == "testFixture"

    def test_get_dimension_value(self, token_replay):
        token_replay.fitness = 0.1
        token_replay.simplicity = 0.2
        token_replay.precision = 0.3
        token_replay.generalization = 0.5

        assert 0.1 == token_replay.get_dimension_value("f")
        assert 0.1 == token_replay.get_dimension_value("fitness")

        assert 0.2 == token_replay.get_dimension_value("s")
        assert 0.2 == token_replay.get_dimension_value("simplicity")

        assert 0.3 == token_replay.get_dimension_value("p")
        assert 0.3 == token_replay.get_dimension_value("precision")

        assert 0.5 == token_replay.get_dimension_value("g")
        assert 0.5 == token_replay.get_dimension_value("generalization")

        with pytest.raises(ValueError):
            token_replay.get_dimension_value("nonsense_dimension")

    def test_calculate_fitness(self, token_replay):
        token_replay.consumed_tokens = 80
        token_replay.produced_tokens = 90
        token_replay.missing_tokens = {"a": 3, "b": 4}
        token_replay.remaining_tokens = {"a": 4}
        expected_fitness = 0.9340277

        fitness = token_replay.calculate_fitness()

        assert isinstance(fitness, float)
        assert 0 <= fitness <= 1

        assert fitness == pytest.approx(expected_fitness, 0.01)

    def test_calculate_pm4py_dimensions(self, token_replay):
        # Run the _calculate_pm4py_dimensions method
        fitness, simplicity, precision, generalization = token_replay._calculate_pm4py_dimensions(
            token_replay.log, token_replay.net, token_replay.initial_marking, token_replay.final_marking
        )

        assert isinstance(fitness, float) or isinstance(fitness, int)
        assert isinstance(simplicity, float) or isinstance(simplicity, int)
        assert isinstance(precision, float) or isinstance(precision, int)
        assert isinstance(generalization, float) or isinstance(generalization, int)

        assert 0 <= fitness <= 1
        assert 0 <= simplicity <= 1
        assert 0 <= precision <= 1
        assert 0 <= generalization <= 1

    @pytest.mark.parametrize(
        "log",
        [
            [],
            [("a",), ("b",)],
            [("a", "b", "c"), ("c", "d", "e"), ("a", "e")]
        ]
    )
    def test_shuffle_activities(self, token_replay, log):
        token_replay.log = log

        # Run the shuffle_activities method
        shuffled_log = token_replay.shuffle_activities()

        # Check if the result is a list
        assert isinstance(shuffled_log, list)

        if not shuffled_log or not any(len(trace) > 1 for trace in shuffled_log):
            return

        # Check if the length of the shuffled log is the same as the original log
        assert len(shuffled_log) == len(token_replay.log)

        # rerun test, if there is any trace, that is different compared to original log
        if not any(shuffled_log[i] != token_replay.log[i] for i in range(len(shuffled_log))):
            self.test_shuffle_activities(token_replay, log)

        # Check if each trace in the shuffled log has same length as corresponding trace in original log
        for i in range(len(shuffled_log)):
            assert len(shuffled_log[i]) == len(token_replay.log[i])

        # Check if each trace in the shuffled log contains same activities as corresponding trace in original log
        for i in range(len(shuffled_log)):
            assert set(shuffled_log[i]) == set(token_replay.log[i])

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