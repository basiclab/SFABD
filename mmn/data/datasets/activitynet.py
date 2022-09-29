import json
import logging
import torch
from .utils import  moment_to_iou2d, bert_embedding, get_vid_feat
from transformers import DistilBertTokenizer

## for drawing
from matplotlib import pyplot as plt

class ActivityNetDataset(torch.utils.data.Dataset):
    def __init__(self, ann_file, feat_file, num_pre_clips, num_clips):
        super(ActivityNetDataset, self).__init__()
        self.feat_file = feat_file
        self.num_pre_clips = num_pre_clips
        with open(ann_file, 'r') as f:
            annos = json.load(f)

        self.annos = []
        self.video_name_to_index = {}
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        logger = logging.getLogger("mmn.trainer")
        logger.info("Preparing data, please wait...")

        for i, (vid, anno) in enumerate(annos.items()):
            self.video_name_to_index[vid] = i
            duration = anno['duration']
            # Produce annotations
            moments = []
            all_iou2d = []
            sentences = []
            for timestamp, sentence in zip(anno['timestamps'], anno['sentences']):
                if timestamp[0] < timestamp[1]:
                    moment = torch.Tensor([max(timestamp[0], 0), min(timestamp[1], duration)])
                    moments.append(moment)
                    iou2d = moment_to_iou2d(moment, num_clips, duration)
                    all_iou2d.append(iou2d)
                    sentences.append(sentence)

                    '''
                    ## Plot the iou 2dmap of each annotation
                    offset = torch.ones(num_clips, num_clips).triu()*0.01 ## for better visualization
                    cm = plt.cm.get_cmap('Reds')
                    fig, ax = plt.subplots()
                    iou_plot = ax.imshow(iou2d+offset, cmap=cm, vmin=0.0, vmax=1.0)
                    bar = plt.colorbar(iou_plot)
                    plt.xlabel('end index')
                    plt.ylabel('start index')
                    plt.title(f"{vid}\n{sentence}")
                    bar.set_label('IoU')
                    plt.show()
                    '''


            moments = torch.stack(moments)
            all_iou2d = torch.stack(all_iou2d)
            queries, word_lens = bert_embedding(sentences, tokenizer)  # padded query of N*word_len, tensor of size = N
            assert moments.size(0) == all_iou2d.size(0)
            assert moments.size(0) == queries.size(0)
            assert moments.size(0) == word_lens.size(0)
            self.annos.append(
                {
                    'vid': vid,
                    'moment': moments,
                    'iou2d': all_iou2d,
                    'sentence': sentences,
                    'query': queries,
                    'wordlen': word_lens,
                    'duration': duration,
                }
             )

    def __getitem__(self, idx):
        ## self.feat_file: feature file path,  self.annos[idx]['vid']: video name 
        feat = get_vid_feat(self.feat_file, self.annos[idx]['vid'], self.num_pre_clips, dataset_name="activitynet")
        #print(f"feat:{feat.shape}") ## [256, 500]
        return feat, self.annos[idx]['query'], self.annos[idx]['wordlen'], self.annos[idx]['iou2d'], self.annos[idx]['moment'], len(self.annos[idx]['sentence']), idx
    
    def __len__(self):
        return len(self.annos)
    
    def get_duration(self, idx):
        return self.annos[idx]['duration']
    
    def get_sentence(self, idx):
        return self.annos[idx]['sentence']
    
    def get_moment(self, idx):
        return self.annos[idx]['moment']
    
    def get_vid(self, idx):
        return self.annos[idx]['vid']

    ## added
    def get_data_by_video_name(self, video_name):
        feat = get_vid_feat(self.feat_file, self.annos[idx]['vid'], self.num_pre_clips, dataset_name="activitynet")
        #print(f"feat:{feat.shape}") ## [256, 500]
        return feat, self.annos[idx]['query'], self.annos[idx]['wordlen'], self.annos[idx]['iou2d'], self.annos[idx]['moment'], len(self.annos[idx]['sentence']), idx

