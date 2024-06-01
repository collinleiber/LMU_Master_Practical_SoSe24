from practical.ProcessMining.group1.task3.inductiveminer import InductiveMiner


class TestInductiveMiner:
    def test_parallel_cut(self):
        log = [('b', 'c'),
               ('c', 'b')]
        miner = InductiveMiner(log)
        parallel_cut = miner._parallel_cut(miner.dfg, miner.start_activities, miner.end_activities)
        assert parallel_cut == [set('b'), set('c')]
        parallel_split = miner._split_log(miner.logs, parallel_cut)
        sublogs = [sorted([('b',), ('b',)]),
                   sorted([('c',), ('c',)])]
        assert all(sorted(sl) in sublogs for sl in parallel_split)

        log = [('b', 'e'),
               ('b', 'e', 'c', 'd', 'b'),
               ('b', 'c', 'e', 'd', 'b'),
               ('b', 'c', 'd', 'e', 'b'),
               ('e', 'b', 'c', 'd', 'b')]
        miner = InductiveMiner(log)
        parallel_cut = miner._parallel_cut(miner.dfg, miner.start_activities, miner.end_activities)
        assert parallel_cut == [set('e'), set('bcd')]
        parallel_split = miner._split_log(miner.logs, parallel_cut)
        sublogs = [sorted([('e',), ('e',), ('e',), ('e',), ('e',)]),
                   sorted([('c', 'd', 'b'), ('b', 'c', 'd'), ('b', 'c', 'd')])]
        assert all(sorted(sl) in sublogs for sl in parallel_split)

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
        loop_split = miner._split_log(miner.logs, loop_cut)
        sublogs = [sorted([('b', 'c'), ('c', 'b'), ('b', 'c'), ('b', 'c'), ('c', 'b'), ('b', 'c'), ('b', 'c'),
                           ('c', 'b'), ('c', 'b'), ('b', 'c'), ('c', 'b')]),
                   sorted([('e', 'f'), ('e', 'f'), ('e', 'f'), ('e', 'f'), ('e', 'f')])]
        assert all(sorted(sl) in sublogs for sl in loop_split)

    def test_fall_through_flower_model(self):
        log = [('a', 'b', 'c', 'd', 'e', 'f', 'g')]
        miner = InductiveMiner(log)
        sublogs = [sorted([('a',)]), sorted([('b',)]), sorted([('c',)]), sorted([('d',)]),
                   sorted([('e',)]), sorted([('f',)]), sorted([('g',)])]
        assert miner.run() == sublogs

