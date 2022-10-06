import json

import torch
from tqdm import tqdm

from src.datasets.charades.base import CharadesBase
from src.utils import moment_to_iou2d, moments_to_iou2d


class DynamicCharades(CharadesBase):
    def __init__(
        self,
        ann_file,
        num_clips,                      # number of final clips for each video
        feat_file,                      # path to feature file
        num_init_clips,                 # number of small basic clips features. ex. 64 for Charades
        weights=[1.0, 0.0, 0.0],        # sample rate for each number of targets
        duration_low=30,                # duration sampling lower bound
        duration_upp=40,                # duration sampling upper bound
        seed=0,
        **dummy,
    ):
        super().__init__(feat_file, num_init_clips)
        self.num_clips = num_clips
        self.weights = torch.Tensor(weights).float() / sum(weights)
        self.duration_low = duration_low
        self.duration_upp = duration_upp
        self.generator = torch.Generator().manual_seed(seed)
        self.annos = self.parse_template(ann_file)
        self.length = 12404

    # override
    def __len__(self):
        return self.length

    def parse_template(self, ann_file):
        with open(ann_file, 'r') as f:
            raw_annos = json.load(f)

        annos = []
        with tqdm(raw_annos.items(), ncols=0, leave=False) as pbar:
            pbar.set_description("DynamicCharades")
            for query, raw_clips in pbar:
                clips = []
                for raw_clip in raw_clips:
                    duration = torch.tensor(raw_clip['duration'])
                    start, end = raw_clip['timestamps']
                    moment = torch.Tensor([max(start, 0), min(end, duration)])
                    if moment[0] < moment[1]:
                        clips.append({
                            'vid': raw_clip['video'],
                            'sentence': raw_clip['sentence'],
                            'moment': moment,
                            'duration': duration,
                        })
                        assert 0 <= moment[0] <= moment[1] <= duration
                if len(clips) > 0:
                    annos.append({
                        'query': query,
                        'clips': clips,
                    })
        return annos

    # override
    def get_anno(self, _, num_targets=None):
        state = torch.random.get_rng_state()
        torch.random.set_rng_state(self.generator.get_state())

        if num_targets is None:
            num_targets = torch.multinomial(self.weights, 1)[0] + 1
        else:
            num_targets = torch.tensor(num_targets)

        # sample query until query string is not `other`
        while True:
            anno_idx = torch.randint(len(self.annos), ())
            anno = self.annos[anno_idx]
            if num_targets == 1 or anno['query'] != 'other':
                break
        # sample clips
        clips = anno['clips']
        clips_idx = torch.randint(len(clips), (num_targets,))
        clips = [clips[idx] for idx in clips_idx]
        min_duration = sum(
            clip['moment'][1] - clip['moment'][0] for clip in clips)

        # sample duration. Note that this is not the final duration, since
        # the individual clip duration is not always enough.
        diff = self.duration_upp - self.duration_low
        duration = torch.rand(()) * diff + self.duration_low
        remain = duration - min_duration

        ratios = torch.rand(num_targets, 2)
        ratios = ratios / ratios.sum()
        vids = []
        timestamps = []
        sents = []
        moments = []
        durations = []
        pre_sum = 0
        for (pre_ratio, pos_ratio), clip in zip(ratios, clips):
            vids.append(clip['vid'])
            sents.append(clip['sentence'])
            durations.append(clip['duration'])
            # padding time
            pre_time = pre_ratio * remain
            pos_time = pos_ratio * remain
            # calculate timestamps
            moment_st, moment_ed = clip['moment']
            video_st = max(moment_st - pre_time, torch.tensor(0.))
            video_ed = min(moment_ed + pos_time, clip['duration'])
            timestamps.append([
                video_st.item(),
                video_ed.item(),
            ])
            # calculate moment
            moments.append([
                pre_sum + (moment_st - video_st),
                pre_sum + (moment_ed - video_st),
            ])
            # update pre_sum
            pre_sum = pre_sum + (video_ed - video_st)
        moments = torch.Tensor(moments)
        duration = pre_sum

        iou2ds = []
        for moment in moments:
            iou2d = moment_to_iou2d(moment, self.num_clips, duration)
            iou2ds.append(iou2d)
        iou2ds = torch.stack(iou2ds)
        iou2d = moments_to_iou2d(moments, self.num_clips, duration)

        self.generator.set_state(torch.random.get_rng_state())
        torch.random.set_rng_state(state)
        return {
            'vids': vids,
            'timestamps': timestamps,
            'query': anno['query'],
            'sents': sents,
            'iou2d': iou2d,
            'iou2ds': iou2ds,
            'num_targets': num_targets,
            'moments': moments,
            'durations': durations,
            'duration': duration,
        }
