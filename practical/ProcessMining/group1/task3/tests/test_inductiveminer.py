from practical.ProcessMining.group1.task3.inductiveminer import InductiveMiner


class TestInductiveMiner:
    def test_parallel_cut(self):
        log = [('b', 'c'),
               ('c', 'b')]
        miner = InductiveMiner(log)
        parallel_cut = miner._parallel_cut(miner.dfg, miner.start_activities, miner.end_activities)
        assert parallel_cut == [set('b'), set('c')]
        parallel_split = miner._split_log(miner.log, parallel_cut)
        assert sorted(parallel_split) == sorted([('b',), ('c',),
                                                 ('c',), ('b',)])

        log = [('b', 'e'),
               ('b', 'e', 'c', 'd', 'b'),
               ('b', 'c', 'e', 'd', 'b'),
               ('b', 'c', 'd', 'e', 'b'),
               ('e', 'b', 'c', 'd', 'b')]
        miner = InductiveMiner(log)
        parallel_cut = miner._parallel_cut(miner.dfg, miner.start_activities, miner.end_activities)
        assert parallel_cut == [set('e'), set('bcd')]
        parallel_split = miner._split_log(miner.log, parallel_cut)
        assert sorted(parallel_split) == sorted([('e',),
                                                 ('e',), ('c', 'd', 'b'),
                                                 ('e',),
                                                 ('b', 'c', 'd'), ('e',),
                                                 ('e',), ('b', 'c', 'd')])

    def test_loop_cut(self):
        log = [('b', 'c'),
               ('c', 'b'),
               ('b', 'c', 'e', 'f', 'b', 'c'),
               ('c', 'b', 'e', 'f', 'b', 'c'),
               ('b', 'c', 'e', 'f', 'c', 'b'),
               ('c', 'b', 'e', 'f', 'b', 'c', 'e', 'f', 'c', 'b')]
        miner = InductiveMiner(log)
        loop_cut = miner._loop_cut(miner.dfg, miner.start_activities, miner.end_activities)
        assert loop_cut == [set('bc'), set('ef')]
        loop_split = miner._split_log(miner.log, loop_cut)
        assert sorted(loop_split) == sorted([('b', 'c'),
                                             ('c', 'b'),
                                             ('b', 'c'), ('e', 'f'), ('b', 'c'),
                                             ('c', 'b'), ('e', 'f'), ('b', 'c'),
                                             ('b', 'c'), ('e', 'f'), ('c', 'b'),
                                             ('c', 'b'), ('e', 'f'), ('b', 'c'), ('e', 'f'), ('c', 'b')])

    def test_fall_through_flower_model(self):
        log = [('a', 'b', 'c', 'd', 'e', 'f', 'g')]
        miner = InductiveMiner(log)
        miner.run()
        assert sorted(miner.log) == sorted([('a',), ('b',), ('c',), ('d',), ('e',), ('f',), ('g',)])
