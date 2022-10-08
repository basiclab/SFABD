from src.datasets.charades import Charades


class TACoS(Charades):
    # override
    def get_duration(self, anno):
        return anno["num_frames"] / anno["fps"]
