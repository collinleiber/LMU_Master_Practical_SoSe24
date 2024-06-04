from inductiveminer import InductiveMiner


class InductiveMinerInfrequent(InductiveMiner):
    def __init__(self, threshold: float):
        super().__init__()
        self.threshold = threshold

    def get_frequent_directly_follows_graph(self):
        pass

    def _calculate_eventual_follows_graph(self):
        pass

    def get_frequent_eventually_follows_graph(self):
        pass

    def xor_split_infrequent(self):
        pass

    def sequence_split_infrequent(self):
        pass

    def loop_split_infrequent(self):
        pass

