from src.datasets.charades.dynamic import DynamicCharades


class StaticCharades(DynamicCharades):
    def __init__(
        self,
        ann_file,
        num_clips,                      # number of final clips for each video
        feat_file,                      # path to feature file
        num_init_clips,                 # number of small basic clips features. ex. 64 for Charades
        weights=[0.5, 0.4, 0.1],        # sample rate for each number of targets
        duration_low=30,                # duration sampling lower bound
        duration_upp=40,                # duration sampling upper bound
        seed=0,
        **dummy,
    ):
        super().__init__(
            ann_file,
            num_clips,
            feat_file,
            num_init_clips,
            weights,
            duration_low,
            duration_upp,
            seed
        )
        nums = (self.weights * self.length).round().long()
        self.length = nums.sum().item()

        annos = []
        for i, num in enumerate(nums):
            num_targets = i + 1
            for _ in range(num):
                anno = super().get_anno(_, num_targets)
                annos.append(anno)
        self.annos = annos

    # override
    def get_anno(self, idx, num_targets=None):
        return self.annos[idx]

    # override
    def __len__(self):
        return len(self.annos)
