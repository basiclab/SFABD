import os
from os.path import join, exists
import h5py
import numpy as np

import torch
import torchtext
from torch.functional import F

## for plotting
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def iou(candidates, gt):
    start, end = candidates[:,0], candidates[:,1]
    s, e = gt[0].float(), gt[1].float()
    # print(s.dtype, start.dtype)
    inter = end.min(e) - start.max(s)
    union = end.max(e) - start.min(s)
    return inter.clamp(min=0) / union

def box_iou(boxes1, boxes2):
    area1 = box_length(boxes1)
    area2 = box_length(boxes2)
    max_start = torch.max(boxes1[:, None, 0], boxes2[:, 0])  # [N,M]
    min_end = torch.min(boxes1[:, None, 1], boxes2[:, 1])  # [N,M]
    inter = (min_end - max_start).clamp(min=0)  # [N,M]
    union = area1[:, None] + area2 - inter
    iou = inter / union
    return iou


def box_length(boxes):
    return boxes[:, 1] - boxes[:, 0]


## turn the 2d map score into moments([start, end]...) and scores(matching score)
def score2d_to_moments_scores(score2d, num_clips, duration):
    ## num_clips: 64 for activitynet
    grids = score2d.nonzero() ## [64*64, 2] ## return 2d map coordinate of nonzero value
    scores = score2d[grids[:, 0], grids[:, 1]] ## [64*64]
    grids[:, 1] += 1
    moments = grids * duration / num_clips ## all grid moments start, end time, [64*64, 2]
    return moments, scores


## moment: ex. [4.0, 8.4]
def moment_to_iou2d(moment, num_clips, duration):
    iou2d = torch.ones(num_clips, num_clips) ## ex. 16*16 for charades_sta
    candidates, _ = score2d_to_moments_scores(iou2d, num_clips, duration)
    iou2d = iou(candidates, moment).reshape(num_clips, num_clips)
    return iou2d ## the iou GT map

def plot_combined_iou2d(iou2d_list, num_clips, anno):
    fig, axs = plt.subplots(1, len(iou2d_list)+1, figsize=(5*(len(iou2d_list)+1), 5.5))
    offset = torch.ones(num_clips, num_clips).triu()*0.05 ## for better visualization
    cm = plt.cm.get_cmap('Reds')

    combined_iou2d = torch.zeros(num_clips, num_clips)
    for i, iou2d in enumerate(iou2d_list):
        ## plot the original iou2d
        iou_plot = axs[i].imshow(iou2d+offset, cmap=cm, vmin=0.0, vmax=1.0)
        axs[i].set(xlabel='end index', ylabel='start index')
        axs[i].set_title(f"{anno['video'][i]}:  {anno['sentences'][i]}")     
        divider = make_axes_locatable(axs[i])
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(iou_plot, cax=cax)

        ## summing up for combined_iou2d
        remain_iou2d_list = [x for j, x in enumerate(iou2d_list) if j != i]
        for remain_iou2d in remain_iou2d_list:
            iou2d = iou2d - remain_iou2d
        iou2d = iou2d.clamp(min=0.0, max=1)
        iou2d = (iou2d > 0.3) * iou2d
        combined_iou2d += iou2d

    ## plot the combined_iou2d
    iou_plot = axs[-1].imshow(combined_iou2d+offset, cmap=cm, vmin=0.0, vmax=1.0)
    axs[-1].set(xlabel='end index', ylabel='start index')
    axs[-1].set_title("Combined iou2d")
    divider = make_axes_locatable(axs[-1])
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(iou_plot, cax=cax)    

    ## settings
    fig.suptitle(f"{anno['query']}", y=0.9)
    fig.tight_layout()
    plt.show()


## important
## Step 1. generate iou2d for each label
## Step 2. iterate each iou2d map, substract the rest iou2d maps from it, clamp negative value, and only keep clips that > threshold (make the combined iou map has clean boundary)
def multi_moments_to_iou2d(moments, num_clips, duration, anno):
    iou2d_list = []
    for i, moment in enumerate(moments):
        iou2d = torch.ones(num_clips, num_clips) ## ex. 16*16 for charades_sta
        candidates, _ = score2d_to_moments_scores(iou2d, num_clips, duration)
        iou2d = iou(candidates, moment).reshape(num_clips, num_clips)
        iou2d_list.append(iou2d)

    ## merging the iou2d_list
    combined_iou2d = torch.zeros(num_clips, num_clips)
    for i, iou2d in enumerate(iou2d_list):
        remain_iou2d_list = [x for j, x in enumerate(iou2d_list) if j != i]
        for remain_iou2d in remain_iou2d_list:
            iou2d = iou2d - remain_iou2d
        iou2d = iou2d.clamp(min=0.0, max=1)
        iou2d = (iou2d > 0.5) * iou2d
        combined_iou2d += iou2d

    ## plot the iou2d and combined_iou2d
    #plot_combined_iou2d(iou2d_list, num_clips, anno)

    return combined_iou2d


## Activitynet NUM_PRE_CLIPS: 256
## Charades    NUM_PRE_CLIPS: 32 -> change to 64 now
def avgfeats(feats, num_pre_clips):
    # Produce the feature of per video into fixed shape (e.g. 64 x 4096)
    # Input Example: feats (torch.tensor, ?x4096); num_pre_clips (64)
    num_src_clips = feats.size(0)
    idxs = torch.arange(0, num_pre_clips+1, 1.0) / num_pre_clips * num_src_clips
    idxs = idxs.round().long().clamp(max=num_src_clips-1)
    # To prevent a empty selection, check the idxs
    meanfeats = []
    for i in range(num_pre_clips):
        s, e = idxs[i], idxs[i+1]
        if s < e:
            meanfeats.append(feats[s:e].mean(dim=0))
        else:
            meanfeats.append(feats[s])
    return torch.stack(meanfeats)



def maxfeats(feats, num_pre_clips):
    # Produce the feature of per video into fixed shape (e.g. 256*4096)
    # Input Example: feats (torch.tensor, ?x4096); num_pre_clips (256)
    num_src_clips = feats.size(0)
    idxs = torch.arange(0, num_pre_clips+1, 1.0) / num_pre_clips * num_src_clips
    idxs = idxs.round().long().clamp(max=num_src_clips-1)
    # To prevent a empty selection, check the idxs
    maxfeats = []
    for i in range(num_pre_clips):
        s, e = idxs[i], idxs[i+1]
        if s < e:
            maxfeats.append(feats[s:e].max(dim=0)[0])
        else:
            maxfeats.append(feats[s])
    return torch.stack(maxfeats)


## not used anymore
def video2feats(feat_file, vids, num_pre_clips, dataset_name):
    assert exists(feat_file)
    vid_feats = {}
    with h5py.File(feat_file, 'r') as f:
        for vid in vids:
            if dataset_name == "activitynet":
                feat = f[vid]['c3d_features'][:]
            else:
                feat = f[vid][:]
            feat = F.normalize(torch.from_numpy(feat), dim=1)
            vid_feats[vid] = avgfeats(feat, num_pre_clips) 

    return vid_feats

## get VGG feat
def get_vid_feat(feat_file, vid, num_pre_clips, dataset_name):
    assert exists(feat_file)
    with h5py.File(feat_file, 'r') as f:
        if dataset_name == "activitynet":
            feat = f[vid]['c3d_features'][:]
            feat = F.normalize(torch.from_numpy(feat), dim=1)
            #print(f"feat:{feat.shape}") ## [seq_len, 500]
        elif dataset_name == "charades":
            feat = f[vid][:]
            feat = F.normalize(torch.from_numpy(feat), dim=1)
            #print(f"charades feat:{feat.shape}") ## [seq_len, 4096]
        else: ## TACoS
            feat = f[vid][:]
            feat = F.normalize(torch.from_numpy(feat), dim=1)
    #print(f"feat:{feat.shape}") ## (seq_len, 4096)
    #print(f"after average:{avgfeats(feat, num_pre_clips).shape}") ## (num_pre_clips=32, 4096)

    return avgfeats(feat, num_pre_clips)

## get C3D feature
def get_c3d_feat(feat_folder, vid, num_pre_clips, dataset_name):
    assert exists(feat_folder)

    c3d_feature = torch.load(f"{feat_folder}/{vid}.pt")
    c3d_feature = F.normalize(c3d_feature, dim=1)

    return avgfeats(c3d_feature, num_pre_clips)


## VGG for Charades
def combine_vid_feat(feat_file, vids, seq_timestamps, num_pre_clips, dataset_name):
    assert exists(feat_file)
    with h5py.File(feat_file, 'r') as f:
        if dataset_name == "charades_combined":
            result_feat = []
            for i, (vid, (seq_start, seq_end)) in enumerate(zip(vids, seq_timestamps)):
                feat = f[vid][seq_start:seq_end] ## np array
                result_feat.append(feat)
                #print(f"{i}, new_feat:{feat.shape}")

            combined_feat = np.vstack(result_feat)
            combined_feat = F.normalize(torch.from_numpy(combined_feat), dim=1)
            #print(f"result_shape:{combined_feat.shape}")
            
    return avgfeats(combined_feat, num_pre_clips)


## C3D for Charades
def combine_c3d_vid_feat(feat_folder, vids, seq_timestamps, num_pre_clips, dataset_name):
    assert exists(feat_folder)
    if dataset_name == "charades_combined":
        result_feat = []
        for i, (vid, (seq_start, seq_end)) in enumerate(zip(vids, seq_timestamps)):
            c3d_feature = torch.load(f"{feat_folder}/{vid}.pt")
            c3d_feature = c3d_feature[seq_start:seq_end]
            #feat = f[vid][seq_start:seq_end] ## np array
            result_feat.append(c3d_feature)
    
        combined_feat = np.vstack(result_feat)
        combined_feat = F.normalize(torch.from_numpy(combined_feat), dim=1)
        #print(f"result_shape:{combined_feat.shape}")
  
    return avgfeats(combined_feat, num_pre_clips)


def get_feat_didemo(feat_file, vid):
    assert exists(feat_file)
    with h5py.File(feat_file, 'r') as f:
        feat = f[vid][:]
    return torch.from_numpy(feat)

def get_c3d_charades(feat_file, num_pre_clips):
    assert exists(feat_file)
    feat = torch.load(feat_file)
    #feat = F.normalize(feat, dim=1)
    return maxfeats(feat, num_pre_clips)


## just tokenizer
def bert_embedding(sentence, tokenizer):
    query_token = tokenizer(sentence, return_tensors="pt", padding=True)
    word_lens = query_token['attention_mask'].sum(dim=1)
    queries = query_token['input_ids']
    return queries, word_lens


def glove_embedding(sentence, vocabs=[], embedders=[]):
    if len(vocabs) == 0:
        vocab = torchtext.vocab.pretrained_aliases["glove.840B.300d"]()
        vocab.itos.extend(['<unk>'])
        vocab.stoi['<unk>'] = vocab.vectors.shape[0]
        vocab.vectors = torch.cat([vocab.vectors, torch.zeros(1, vocab.dim)], dim=0)
        vocabs.append(vocab)
    
    if len(embedders) == 0:
        embedder = torch.nn.Embedding.from_pretrained(vocab.vectors)
        embedders.append(embedder)
    
    vocab, embedder = vocabs[0], embedders[0]
    word_idxs = torch.tensor([vocab.stoi.get(w.lower(), 400000) for w in sentence.split()], dtype=torch.long)
    return embedder(word_idxs)
