import json

import torch
from tqdm import tqdm

from mmn.datasets.charades.static import StaticMultiTargetCharades
from mmn.utils import moment_to_iou2d, moments_to_iou2d


class DynamicMultiTargetCharades(StaticMultiTargetCharades):
    def __init__(
        self,
        ann_file,
        template_file,
        num_clips,                      # number of final clips for each video
        vgg_feat_file,                  # path to feature file
        num_init_clips,                 # number of small basic clips features. ex. 64 for Charades
        weights=[0.5, 0.4, 0.1],        # sample rate for each number of targets
        duration_low=30,                # duration sampling lower bound
        duration_upp=40,                # duration sampling upper bound
        **dummy,
    ):
        super().__init__(ann_file, num_clips, vgg_feat_file, num_init_clips)
        self.num_clips = num_clips
        self.weights = torch.Tensor(weights).float()
        self.duration_low = duration_low
        self.duration_upp = duration_upp
        self.multi_annos = self.parse_template(template_file)

    def parse_template(self, template_file):
        with open(template_file, 'r') as f:
            raw_annos = json.load(f)
        annos = []
        for query, raw_clips in tqdm(raw_annos.items(), ncols=0, leave=False):
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
            if query != 'other' and len(clips) >= 3:
                annos.append({
                    'query': query,
                    'clips': clips,
                })
        return annos

    def get_anno(self, _):
        num_targets = torch.multinomial(self.weights, 1)[0] + 1

        # single target
        if num_targets == 1:
            anno_idx = torch.randint(len(self.annos), ())
            return self.annos[anno_idx]

        # sample query
        multi_anno_idx = torch.randint(len(self.multi_annos), ())
        multi_anno = self.multi_annos[multi_anno_idx]
        # sample clips
        clips = multi_anno['clips']
        clips_idx = torch.randint(len(clips), (num_targets,))
        clips = [clips[idx] for idx in clips_idx]
        min_duration = sum(
            clip['moment'][1] - clip['moment'][0] for clip in clips)

        # sample duration. Note that this is not the final duration, since
        # the individual clip duration is not always enough to
        diff = self.duration_upp - self.duration_low
        duration = torch.rand(()) * diff + self.duration_low
        remain = duration - min_duration

        ratios = torch.rand(num_targets, 2)
        ratios = ratios / ratios.sum()
        vid = []
        seq_index = []
        sents = []
        moments = []
        pre_sum = 0
        for (pre_ratio, pos_ratio), clip in zip(ratios, clips):
            vid.append(clip['vid'])
            sents.append(clip['sentence'])
            # padding time
            pre_time = pre_ratio * remain
            pos_time = pos_ratio * remain
            # calculate seq_index
            moment_st, moment_ed = clip['moment']
            video_st = max(moment_st - pre_time, torch.tensor(0.))
            video_ed = min(moment_ed + pos_time, clip['duration'])
            seq_index.append([
                torch.round(video_st * 6).long().item(),
                torch.round(video_ed * 6).long().item(),
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

        return {
            'vid': vid,
            'seq_index': seq_index,
            'query': multi_anno['query'],
            'sents': sents,
            'iou2d': iou2d,
            'iou2ds': iou2ds,
            'num_targets': num_targets,
            'moments': moments,
            'duration': duration,
        }

    def __len__(self):
        return 12404
