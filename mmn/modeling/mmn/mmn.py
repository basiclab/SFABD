import torch
from torch import nn
from torch.functional import F
from .featpool import build_featpool  # downsample 1d temporal features to desired length
from .feat2d import build_feat2d  # use MaxPool1d/Conv1d to generate 2d proposal-level feature map from 1d temporal features
from .loss import build_contrastive_loss, build_contrastive_loss_multi_labels
from .loss import build_bce_loss
from .text_encoder import build_text_encoder
from .proposal_conv import build_proposal_conv
import numpy as np


class MMN(nn.Module):
    def __init__(self, cfg):
        super(MMN, self).__init__()
        self.only_iou_loss_epoch = cfg.SOLVER.ONLY_IOU
        self.featpool = build_featpool(cfg) 
        self.feat2d = build_feat2d(cfg) ## conv or pool for clip feature
        #self.contrastive_loss = build_contrastive_loss(cfg, self.feat2d.mask2d) ## modify this in the future
        self.contrastive_loss = build_contrastive_loss_multi_labels(cfg, self.feat2d.mask2d) ## modify this in the future
        self.iou_loss = build_bce_loss(cfg, self.feat2d.mask2d)
        self.text_encoder = build_text_encoder(cfg)
        self.proposal_conv = build_proposal_conv(cfg, self.feat2d.mask2d)
        self.joint_space_size = cfg.MODEL.MMN.JOINT_SPACE_SIZE
        self.encoder_name = cfg.MODEL.MMN.TEXT_ENCODER.NAME

    def forward(self, batches, cur_epoch=1):
        """
        Arguments:
            batches.iou2d: list(B) num_sent x T x T
            feat2ds: B x C x T x T
            query_feats: list(B) num_sent x C
        """

        ious2d = batches.iou2d
        separate_iou2d = batches.separate_iou2d  ## used for selecting the top-3 iou proposals with gt

        assert len(ious2d) == batches.feats.size(0) ## batch_size

        
        feats = self.featpool(batches.feats)  # from pre_num_clip to num_clip with overlapped average pooling, e.g., 32 -> 16

        map2d = self.feat2d(feats)  # use MaxPool1d to generate 2d proposal-level feature map from 1d temporal features



        ## ConvNet of 2D temporal map and conv1x1 to project proposals to embedding space
        map2d = self.proposal_conv(map2d) 

        ## query encoder, project to embedding space
        query_feats = self.text_encoder(batches.tokenized_queries, batches.wordlens)

        ## for intra_query loss    
        queries = batches.queries

        num_target_batch = batches.num_target

        ## only keep multi-target samples
        mask = np.array([num > 1 for num in num_target_batch])
        queries = np.array(queries)
        selected_query_templates = queries[mask]
        selected_query_templates = np.expand_dims(selected_query_templates, axis=1)

        ## preparing intra_query_mask to prune false neg samples        
        ## if is same query template, then value will be 0, other neg samples will be 1
        intra_query_mask = np.ones((selected_query_templates.size, selected_query_templates.size))
        ## diagonal line is all 0
        for i in range(selected_query_templates.size):
            for j in range(selected_query_templates.size):
                if selected_query_templates[i] == selected_query_templates[j]:
                    ## mask the sample with same query_template
                    intra_query_mask[i, j] = 0

        ## query feats of original query in multi-target samples
        original_query_feats = self.text_encoder(batches.original_tokenized_queries, batches.original_word_lens)

        # inference
        contrastive_scores = []
        matching_scores = []
        matching_scores_original = []
        _, T, _ = map2d[0].size()

        for i, sf in enumerate(query_feats):  # query_feats: [num_sent x C] (len=B)
            # iou part
            vid_feat = map2d[i]  # C x T x T
            vid_feat_norm = F.normalize(vid_feat, dim=0)
            sf_norm = F.normalize(sf, dim=1)
            matching_score = torch.mm(sf_norm, vid_feat_norm.reshape(vid_feat_norm.size(0), -1)).reshape(-1, T, T)  # num_sent x T x T
            matching_scores.append((matching_score*10).sigmoid() * self.feat2d.mask2d)

        # loss
        if self.training:
            loss_iou = self.iou_loss(torch.cat(matching_scores, dim=0), torch.cat(ious2d, dim=0), cur_epoch)
            loss_vid, loss_sent, loss_intra_vid, loss_intra_query = self.contrastive_loss(map2d, query_feats, ious2d, separate_iou2d, batches.moments, original_query_feats, num_target_batch, intra_query_mask)

            return loss_vid, loss_sent, loss_intra_vid, loss_intra_query, loss_iou

        else:
            return map2d, query_feats, matching_scores  # first two maps for visualization

    
    ## for compute on new video
    def evaluate(self, batches, cur_epoch=1):
        """
        Arguments:
            feat2ds: B x C x T x T
            query_feats: list(B) num_sent x C
        """
        # backbone
        feats = self.featpool(batches.feats)  # from pre_num_clip to num_clip with overlapped average pooling, e.g., 256 -> 128
        map2d = self.feat2d(feats)  # use MaxPool1d to generate 2d proposal-level feature map from 1d temporal features
        map2d = self.proposal_conv(map2d)
        query_feats = self.text_encoder(batches.queries, batches.wordlens)

        # inference
        contrastive_scores = []
        matching_scores = []
        _, T, _ = map2d[0].size()

        for i, sf in enumerate(query_feats):  # sent_feat: [num_sent x C] (len=B)
            # iou part
            vid_feat = map2d[i]  # C x T x T
            vid_feat_norm = F.normalize(vid_feat, dim=0)
            sf_norm = F.normalize(sf, dim=1)
            matching_score = torch.mm(sf_norm, vid_feat_norm.reshape(vid_feat_norm.size(0), -1)).reshape(-1, T, T)  # num_sent x T x T
            matching_scores.append((matching_score*10).sigmoid() * self.feat2d.mask2d)

               
        return map2d, query_feats, matching_scores  # first two maps for visualization
    