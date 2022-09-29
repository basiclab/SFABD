import os
import json
import logging
import torch
from .utils import moment_to_iou2d,  bert_embedding, get_vid_feat, get_c3d_feat, combine_vid_feat, combine_c3d_vid_feat, multi_moments_to_iou2d
from transformers import DistilBertTokenizer

## for online multi-target sample generation
import random
import csv
import pandas as pd



class CharadesDataset(torch.utils.data.Dataset):
    def __init__(self, ann_file, feat_file, c3d_feat_folder, num_pre_clips, num_clips, feat_type):
        super(CharadesDataset, self).__init__()
        self.feat_file = feat_file ## path to feature file
        self.num_pre_clips = num_pre_clips ## number of small basic clips features. 32 for Charades
        with open(ann_file, 'r') as f:
            annos = json.load(f)

        self.feat_type = feat_type
        self.c3d_feat_folder = c3d_feat_folder  ## added
        self.annos = []
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        logger = logging.getLogger("mmn.trainer")
        logger.info("Preparing data, please wait...")

        for vid, anno in annos.items():
            duration = anno['duration'] ## video length
            # Produce annotations
            moments = [] ## len = number of annotations in a video
            all_iou2d = []
            sentences = [] ## all query
            for timestamp, sentence in zip(anno['timestamps'], anno['sentences']): ## may have multiple annotations in a video
                ## ex: time:[4.0, 8.9], sentence:person turns the lights on.
                if timestamp[0] < timestamp[1]: ## start < end, legit time stamp
                    moment = torch.Tensor([max(timestamp[0], 0), min(timestamp[1], duration)]) ## ex. (4.0, 8.9)
                    moments.append(moment)
                    iou2d = moment_to_iou2d(moment, num_clips, duration) ## num_clips: 16, moment: (4.0, 8.9)
                    all_iou2d.append(iou2d)
                    sentences.append(sentence)

            moments = torch.stack(moments)
            all_iou2d = torch.stack(all_iou2d)
            queries, word_lens = bert_embedding(sentences, tokenizer)  # padded query of N*word_len, tensor of size = N
            ## queries: (num_query_per_video, max_word_length) of word_id

            assert moments.size(0) == all_iou2d.size(0)
            assert moments.size(0) == queries.size(0)
            assert moments.size(0) == word_lens.size(0)

            self.annos.append(
                {
                    'vid': vid,
                    'moment': moments,
                    'iou2d': all_iou2d,   ## GT for bce_loss
                    'sentence': sentences,
                    'query': queries,     ## Bert tokenizer output
                    'wordlen': word_lens, ## Bert tokenizer output
                    'duration': duration,
                }
            )

        #self.feats = video2feats(feat_file, annos.keys(), num_pre_clips, dataset_name="charades")

    def __getitem__(self, idx):
        #feat = self.feats[self.annos[idx]['vid']]

        if self.feat_type == "VGG":
            feat = get_vid_feat(self.feat_file, self.annos[idx]['vid'], self.num_pre_clips, dataset_name="charades") ## (seq_len, 4096) -> (32, 4096)
        #elif self.feat_type == "C3D":
        #    feat = get_c3d_feat(self.c3d_feat_folder, self.annos[idx]['vid'], self.num_pre_clips, dataset_name="charades") ## (seq_len, 4096) -> (32, 4096)
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


class CharadesCombinedDataset(torch.utils.data.Dataset):
    def __init__(self, ann_file, feat_file, c3d_feat_folder, num_pre_clips, num_clips, feat_type):
        super(CharadesCombinedDataset, self).__init__()
        self.feat_file = feat_file ## path to feature file
        self.num_pre_clips = num_pre_clips ## number of small basic clips features. 32 for Charades (now 64)
        with open(ann_file, 'r') as f:
            annos = json.load(f)

        self.feat_type = feat_type
        self.c3d_feat_folder = c3d_feat_folder  ## added
        self.annos = []
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        logger = logging.getLogger("mmn.trainer")
        logger.info("Preparing data, please wait...")

        for sample_id, anno in annos.items():
            duration = anno['duration'] ## video length
            # Produce annotations
            moments = [] ## len = number of annotations in a video


            original_queries = [] ## all query from multiple samples
            separate_iou2d = []
            for timestamp, original_query in zip(anno['timestamps'], anno['sentences']): ## may have multiple annotations in a video
                ## ex: time:[4.0, 8.9], sentence:person turns the lights on.
                if timestamp[0] < timestamp[1]: ## start < end, legit time stamp
                    moment = torch.Tensor([max(timestamp[0], 0), min(timestamp[1], duration)]) ## ex. (4.0, 8.9)
                    moments.append(moment)
                    iou2d = moment_to_iou2d(moment, num_clips, duration) ## num_clips: 16, moment: (4.0, 8.9)
                    separate_iou2d.append(iou2d)
                    original_queries.append(original_query)

            ## testing (will plot iou2d graph)
            ## One combined iou2d for each sample
            combined_iou2d = multi_moments_to_iou2d(moments, num_clips, duration, anno)
            combined_iou2d = torch.stack([combined_iou2d])
            moments = torch.stack(moments) ## used for evaluation, computing iou

            ## padded query of N*word_len, tensor of size = N
            ## bert_embedding is just tokenizer
            query_token, word_len = bert_embedding([anno['query']], tokenizer) ## one query per sample

            original_queries_token, original_word_lens = bert_embedding(original_queries, tokenizer)


            self.annos.append(
                {
                    'vid': anno['video'],
                    'seq_index': anno['seq_index'],
                    'moment': moments,
                    'iou2d': combined_iou2d, ## GT for bce_loss
                    'separate_iou2d': separate_iou2d, ## list of separate iou2d
                    'query': anno['query'], ## list of original query???
                    'query_token': query_token, ## query emb
                    'wordlen': word_len,
                    'duration': duration,
                    ## added for intra-query loss
                    'original_queries': original_queries, ## original sample sentences
                    'original_tokenized_queries': original_queries_token,
                    'original_word_lens': original_word_lens,
                }
            )


    def __getitem__(self, idx):
        rand = random.random()
        if self.feat_type == "VGG":
            feat = combine_vid_feat(self.feat_file, self.annos[idx]['vid'], self.annos[idx]['seq_index'], self.num_pre_clips, dataset_name="charades_combined")

        #feats, sentences, tokenized_queries, wordlens, ious2d, separate_iou2d, moments, num_sentence, all_sentences, all_tokenized_query, all_word_len, idxs = transposed_batch
        return (
            feat,

            ## query
            self.annos[idx]['query'], ## used in multi-target sample computing intra_query loss
            self.annos[idx]['query_token'], 
            self.annos[idx]['wordlen'],

            ## iou map
            self.annos[idx]['iou2d'],  ## combined iou2d
            self.annos[idx]['separate_iou2d'], ## each target's iou2d
            
            ## gt timestamps
            self.annos[idx]['moment'],
            
            ## num_target
            len(self.annos[idx]['vid']), 
            
            ## 
            self.annos[idx]['original_queries'],
            self.annos[idx]['original_tokenized_queries'],
            self.annos[idx]['original_word_lens'],
            idx,
        )


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

    def get_iou2d(self, idx):
        return self.annos[idx]['iou2d']

    def get_separate_iou2d(self, idx):
        return self.annos[idx]['separate_iou2d']

###########################################################################################
'''
def decide_clip_boundary(anno, target_clip_length):
    GT_start, GT_end = anno['timestamps']
    available_start = [max(GT_end - target_clip_length, 0), GT_start]
    if available_start[0] > GT_start: ## GT_length longer than target_clip_length
        target_clip_length = target_clip_length + (available_start[0]-GT_start)
        available_start[0] = GT_start

    clip_start = round(random.uniform(available_start[0], available_start[1]), 1)
    clip_end = round(min(clip_start + target_clip_length, anno['duration']), 1)
    return [clip_start, clip_end]


## Charades vgg feature: encode video at 24 fps and extract vgg fc7 feature of 0001.jpg, 0005.jpg, 0009.jpg, 0013.jpg, 0017.jpg, 0021.jpg, 0025.jpg..... 
def convert_time_to_seq_index(time_stamp):
    ## VGG feature used in 2DTAN convert 1s of video frames into [6, 4096] video feature
    start, end = time_stamp
    start_seq_index = round(start * 6)
    end_seq_index = round(end * 6)

    return [start_seq_index, end_seq_index]


## 24 fps, extract 6 frames in one second
def convert_seq_index_to_time(seq_index):
    start_index, end_index = seq_index
    start_time = start_index / 6
    end_time = end_index / 6
    
    return [float(start_time), float(end_time)]


class CharadesCombinedDataset_online_augmentation(torch.utils.data.Dataset):
    def __init__(self, ann_file, feat_file, c3d_feat_folder, num_pre_clips, num_clips, feat_type):
        super(CharadesCombinedDataset, self).__init__()
        self.feat_file = feat_file ## path to feature file
        self.num_pre_clips = num_pre_clips ## number of small basic clips features. 32 for Charades (now 64)
        with open(ann_file, 'r') as f:
            annos = json.load(f)

        self.feat_type = feat_type
        self.c3d_feat_folder = c3d_feat_folder  ## added
        self.annos = []

        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

        ## load the vp_count_train_new, for computing class sampling probability
        self.vp_count_dict = {}
        self.vp_count_df = pd.read_csv(f"/home/vin30731/Desktop/TLG_Dataset_Preprocessing/train/vp_count_train_new.csv")
        for i, row in self.vp_count_df.iterrows():
            if row['Query Template'] != 'other' and row['Sample count'] >= 2:
                self.vp_count_dict[row['Query Template']] = row['Sample count']


        ## load the vp_group_train_new, for combining samples
        with open('/home/vin30731/Desktop/TLG_Dataset_Preprocessing/train/vp_group_train_new.json') as f:
            self.vp_samples_dict = json.load(f)


        self.multi_target_avg_video_length = 45


        logger = logging.getLogger("mmn.trainer")
        logger.info("Preparing data, please wait...")
        for sample_id, anno in annos.items():
            duration = anno['duration'] ## video length
            # Produce annotations
            moments = [] ## len = number of annotations in a video
            sentences = [] ## all query from multiple samples
            separate_iou2d = []
            for timestamp, sentence in zip(anno['timestamps'], anno['sentences']): ## may have multiple annotations in a video
                ## ex: time:[4.0, 8.9], sentence:person turns the lights on.
                if timestamp[0] < timestamp[1]: ## start < end, legit time stamp
                    moment = torch.Tensor([max(timestamp[0], 0), min(timestamp[1], duration)]) ## ex. (4.0, 8.9)
                    moments.append(moment)
                    iou2d = moment_to_iou2d(moment, num_clips, duration) ## num_clips: 16, moment: (4.0, 8.9)
                    separate_iou2d.append(iou2d)
                    sentences.append(sentence)

            ## testing (will plot iou2d graph)
            ## One combined iou2d for each sample
            combined_iou2d = multi_moments_to_iou2d(moments, num_clips, duration, anno)
            combined_iou2d = torch.stack([combined_iou2d])
            moments = torch.stack(moments) ## used for evaluation, computing iou

            ## padded query of N*word_len, tensor of size = N
            ## change to one query for each sample
            ## bert_embedding is just tokenizer
            query, word_len = bert_embedding([anno['query']], tokenizer) ## one query per sample
            all_sentences = sentences
            all_queries, all_word_lens = bert_embedding(all_sentences, tokenizer)


            self.annos.append(
                {
                    'vid': anno['video'],
                    'seq_index': anno['seq_index'],
                    'moment': moments,
                    'iou2d': combined_iou2d, ## GT for bce_loss
                    'separate_iou2d': separate_iou2d, ## list of separate iou2d
                    'sentence': [anno['query']], ## original query template 
                    'query': query, ## query emb
                    'wordlen': word_len,
                    'duration': duration,

                    ## added for intra-query loss
                    'all_sentences': all_sentences, ## original sample sentences
                    'all_tokenized_queries': all_queries,
                    'all_word_lens': all_word_lens,
                }
            )


    def __getitem__(self, idx):
        rand = random.random()

        ## 50% probability use 1-gt samples, do the same thing as before
        if rand <= 0.5:
            if self.feat_type == "VGG":
                feat = combine_vid_feat(self.feat_file, self.annos[idx]['vid'], self.annos[idx]['seq_index'], self.num_pre_clips, dataset_name="charades_combined")

            #feats, sentences, tokenized_queries, wordlens, ious2d, separate_iou2d, moments, num_sentence, all_sentences, all_tokenized_query, all_word_len, idxs = transposed_batch
            return feat, self.annos[idx]['sentence'], self.annos[idx]['query'], self.annos[idx]['wordlen'], self.annos[idx]['iou2d'], self.annos[idx]['separate_iou2d'], self.annos[idx]['moment'], \
                   len(self.annos[idx]['all_sentences']), self.annos[idx]['all_sentences'], self.annos[idx]['all_tokenized_queries'], self.annos[idx]['all_word_lens'], idx

        ## 50% probability online combining multi-target samples
        elif rand > 0.5:
            ## 2-target sample 
            if rand <= 0.9:
                num_target = 2

            ## 3-target sample
            else:
                num_target = 3
                        
            ## sample a query template to combine sample
            ## use equal weight for now
            query_template = random.choice(list(self.vp_count_dict.keys()))
            combinable_samples = self.vp_samples_dict[query_template]  ## combinable samples of the query_template
            clip_length = self.multi_target_avg_video_length / num_target ## clip length for each individual target
            
            ## sampling the index 
            selected_index = random.sample(range(0, length(combinable_samples)))

            ## combine new sample and record the annotation
            new_length = 0
            new_anno = {}
            new_anno['video'] = []
            new_anno['timestamps'] = [] ## new video GT time stamp
            new_anno['seq_index'] = [] ## seq_index of the original vgg feature
            #new_anno['c3d_seq_index'] = [] ## for c3d
            new_anno['sentences'] = []
            new_anno['query'] = query_template        

            for index in selected_index:
                anno = combinable_samples[index]

                ## random select video clip boundary (always contains gt clip)
                clip_start, clip_end = decide_clip_boundary(anno, clip_length)
                seq_start, seq_end = convert_time_to_seq_index([clip_start, clip_end])
                new_anno['seq_index'].append([seq_start, seq_end])

                ## clip_start, clip_end: the boundary of cut clip from original sample video
                ## anno['timestamps']: the gt moment of the original sample video
                new_anno_start = round(anno['timestamps'][0] - clip_start + new_length, 1)
                new_anno_end = round(anno['timestamps'][1] - clip_start + new_length, 1)
                new_anno['timestamps'].append([new_anno_start, new_anno_end])
                new_anno['video'].append(anno['video'])
                new_anno['sentences'].append(anno['sentences'])

                new_length += (clip_end - clip_start)

            new_anno['duration'] = round(new_length, 1)

            feat = combine_vid_feat(self.feat_file, new_anno['video'], new_anno['seq_index'], self.num_pre_clips, dataset_name="charades_combined")

            ## iou2d

            ## separate_iou_2d

            ## moments



            #feats, sentences, tokenized_queries, wordlens, ious2d, separate_iou2d, moments, num_sentence, all_sentences, all_tokenized_query, all_word_len, idxs = transposed_batch
            return feat, [new_anno['query']], self.annos[idx]['query'], self.annos[idx]['wordlen'], \
                    self.annos[idx]['iou2d'], self.annos[idx]['separate_iou2d'], self.annos[idx]['moment'], \
                    len(new_anno['sentences']), self.annos[idx]['all_sentences'], \
                    self.annos[idx]['all_tokenized_queries'], self.annos[idx]['all_word_lens'], idx ## idx is not used


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

    def get_iou2d(self, idx):
        return self.annos[idx]['iou2d']

    def get_separate_iou2d(self, idx):
        return self.annos[idx]['separate_iou2d']
'''