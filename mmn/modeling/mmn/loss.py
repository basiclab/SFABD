import torch
from torch.functional import F
from mmn.data.datasets.utils import box_iou

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from itertools import combinations
import numpy as np


class BceLoss(object):
    def __init__(self, min_iou, max_iou, mask2d):
        self.min_iou, self.max_iou = min_iou, max_iou
        self.mask2d = mask2d
        self.bceloss = torch.nn.BCELoss(reduction='none')
        self.hinge_loss = False

    def linear_scale(self, iou):
        return (iou - self.min_iou) / (self.max_iou - self.min_iou)

    def __call__(self, scores2d, ious2d, epoch):
        iou1d = ious2d.masked_select(self.mask2d)
        scores1d = scores2d.masked_select(self.mask2d)
        loss = 0
        iou1d = self.linear_scale(iou1d).clamp(0, 1)
        loss += self.bceloss(scores1d, iou1d).mean()
        return loss


def build_bce_loss(cfg, mask2d):
    min_iou = cfg.MODEL.MMN.LOSS.MIN_IOU 
    max_iou = cfg.MODEL.MMN.LOSS.MAX_IOU
    return BceLoss(min_iou, max_iou, mask2d)


class ContrastiveLoss(object):
    def __init__(self, cfg, mask2d):
        self.mask2d = mask2d
        self.T_v = cfg.MODEL.MMN.LOSS.TAU_VIDEO ## 0.1
        self.T_s = cfg.MODEL.MMN.LOSS.TAU_SENT ## 0.1
        self.cri = torch.nn.CrossEntropyLoss()
        self.neg_iou = cfg.MODEL.MMN.LOSS.NEGATIVE_VIDEO_IOU ## 0.5
        self.top_k = cfg.MODEL.MMN.LOSS.NUM_POSTIVE_VIDEO_PROPOSAL ## 1
        self.sent_removal_iou = cfg.MODEL.MMN.LOSS.SENT_REMOVAL_IOU ## 0.5
        self.margin = cfg.MODEL.MMN.LOSS.MARGIN ## 0.4
        self.eps = 1e-6
        self.dataset = cfg.DATASETS.NAME

    ## calculate contrastive loss
    ## map2d, sent_feat, ious2d, batches.moments
    def __call__(self, feat2ds, sent_feats, iou2ds, gt_proposals):
        """
            feat2ds: B x C x T x T (map2d)
            sent_feats: list(B) num_sent x C (sent_feat)
            iou2ds: list(B) num_sent x T x T ## GT
            gt_proposals: list(B) num_sent x 2, with format [start, end], unit being seconds (frame/fps)
        """
        # prepare tensors
        B, C, _, _ = feat2ds.size()
        feat1ds = feat2ds.masked_select(self.mask2d).reshape(B, C, -1) ## reshape the 2d map upper triangle to 1d sequence of feature
        feat1ds_norm = F.normalize(feat1ds, dim=1)  # B x C x num_sparse_selected_proposal
        sent_feat_cat = torch.cat(sent_feats, 0)  # sum(num_sent) x C, whole batch
        sum_num_sent = sent_feat_cat.size(0)
        sent_feat_cat_norm = F.normalize(sent_feat_cat, dim=1)  # sum(num_sent) x C, whole batch
        sent_mask = torch.ones(sum_num_sent, sum_num_sent, device=feat2ds.device)

        all_num_sent = [0]
        curr_num_sent = 0
        for i in range(len(sent_feats)): ## iterate a batch
            curr_num_sent += sent_feats[i].size(0)
            all_num_sent.append(curr_num_sent)

        for i, gt_per_video in enumerate(gt_proposals):
            iou_map_per_video = box_iou(gt_per_video, gt_per_video)
            iou_mask = iou_map_per_video < self.sent_removal_iou  # remove high iou sentence, keep low iou sentence
            sent_mask[all_num_sent[i]:all_num_sent[i+1], all_num_sent[i]:all_num_sent[i+1]] = iou_mask.float()
        sent_mask += torch.diag(torch.ones(sum_num_sent, device=feat2ds.device)).float()  # add the sentence itself to the denominator in the loss
        margin_mask = torch.diag(torch.ones(sum_num_sent, device=feat2ds.device)) * self.margin
        vid_pos_list = []
        vid_neg_list = []
        sent_pos_list = []
        sent_neg_list = []

        for i, (sent_feat, iou2d) in enumerate(zip(sent_feats, iou2ds)):  # each video in the batch
            # select positive samples
            #num_sent_this_sample = sent_feat.size(0)
            feat1d = feat1ds_norm[i, :, :]                                                                          # C x num_sparse_selected_proposal
            sent_feat = F.normalize(sent_feat, dim=1)                                                               # num_sent x C
            iou1d = iou2d.masked_select(self.mask2d).reshape(sent_feat.size(0), -1)                                 # num_sent x num_sparse_selected_proposal
            topk_index = torch.topk(iou1d, self.top_k, dim=-1)[1]                                                   # num_sent x top_k
            selected_feat = feat1d.index_select(dim=1, index=topk_index.reshape(-1)).reshape(C, -1, self.top_k)     # C x num_sent x top_k
            selected_feat = selected_feat.permute(1, 2, 0)                                                          # num_sent x top_k x C
            # positive video proposal with pos/neg sentence samples
            vid_pos = torch.bmm(selected_feat,
                                sent_feat.unsqueeze(2)).reshape(-1, self.top_k) - self.margin                       # num_sent x top_k, bmm of (num_sent x top_k x C) and (num_sent x C x 1)
            vid_neg = torch.mm(selected_feat.view(-1, C),
                               sent_feat_cat_norm.t()).reshape(-1, self.top_k, sum_num_sent)                        # num_sent x topk x sum(num_sent), mm of (num_sent*top_k x C) and (C x sum(num_sent))
            vid_pos_list.append(vid_pos)
            vid_neg_list.append(vid_neg)
            # positive sentence with pos/neg video proposals
            sent_pos_list.append(vid_pos.clone())
            sent_neg_same_video = torch.mm(sent_feat, feat1d)                                                   # num_sent x num_sparse_selected_proposal
            iou_neg_mask = (iou1d < self.neg_iou).float()                                                       # only keep the low iou proposals as negative samples in the same video
            sent_neg_same_video = iou_neg_mask * sent_neg_same_video                           # num_sent x num_sparse_selected_proposal
            feat1d_other_video = feat1ds_norm.index_select(dim=0, index=torch.arange(
                B, device=feat2ds.device)[torch.arange(B, device=feat2ds.device) != i])                         # (B-1) x C x num_sparse_selected_proposal
            feat1d_other_video = feat1d_other_video.transpose(1, 0).reshape(C, -1)                              # C x ((B-1) x num_sparse_selected_proposal)
            sent_neg_other_video = torch.mm(sent_feat, feat1d_other_video)                                      # num_sent x ((B-1) x num_sparse_selected_proposal)
            sent_neg_all = [vid_pos.clone().unsqueeze(2),
                            sent_neg_same_video.unsqueeze(1).repeat(1, self.top_k, 1),
                            sent_neg_other_video.unsqueeze(1).repeat(1, self.top_k, 1)]
            sent_neg_list.append(torch.cat(sent_neg_all, dim=2))                                # num_sent x topk x (1 + num_same + num_other)
        vid_pos = (torch.cat(vid_pos_list, dim=0).transpose(0, 1)) / self.T_v                   # top_k x num_sent
        vid_neg = torch.cat(vid_neg_list, dim=0).permute(1, 0, 2)                               # top_k x this_cat_to_be_sum(num_sent) x sum(num_sent)
        vid_neg = (vid_neg - margin_mask) / self.T_v                                            # top_k x this_cat_to_be_sum(num_sent) (positive) x sum(num_sent) (negative)
        sent_mask += torch.diag(torch.ones(sum_num_sent, device=feat2ds.device)).float()
        vid_neg_exp = torch.exp(vid_neg) * sent_mask.clamp(min=0, max=1)
        loss_vid = -(vid_pos - torch.log(vid_neg_exp.sum(dim=2, keepdim=False))).mean()
        sent_pos = torch.cat(sent_pos_list, dim=0) / self.T_s
        sent_neg = torch.cat(sent_neg_list, dim=0) / self.T_s
        sent_neg_exp = torch.exp(sent_neg)
        loss_sent = -(sent_pos - torch.log(sent_neg_exp.sum(dim=2, keepdim=False) + self.eps)).mean()
        return loss_vid, loss_sent


def build_contrastive_loss(cfg, mask2d):
    return ContrastiveLoss(cfg, mask2d)


class ContrastiveLoss_multi_labels(object):
    def __init__(self, cfg, mask2d):
        self.mask2d = mask2d
        self.T_v = cfg.MODEL.MMN.LOSS.TAU_VIDEO ## 0.1
        self.T_s = cfg.MODEL.MMN.LOSS.TAU_SENT ## 0.1
        self.cri = torch.nn.CrossEntropyLoss()
        self.neg_iou = cfg.MODEL.MMN.LOSS.NEGATIVE_VIDEO_IOU ## 0.5
        self.top_k = cfg.MODEL.MMN.LOSS.NUM_POSTIVE_VIDEO_PROPOSAL ## 1
        self.sent_removal_iou = cfg.MODEL.MMN.LOSS.SENT_REMOVAL_IOU ## 0.5
        self.margin = cfg.MODEL.MMN.LOSS.MARGIN ## 0.4
        self.eps = 1e-6
        self.dataset = cfg.DATASETS.NAME

    ## calculate contrastive loss
    ## map2d, sent_feat, ious2d, batches.moments
    def __call__(self, feat2ds, sent_feats, iou2ds, separate_iou2ds, gt_proposals, original_sent_feats, num_sentences, intra_query_mask):
        """
            num_sent is always 1
            feat2ds: B x C x T x T (map2d)
            sent_feats: list(B) 1 x C (sent_feat)
            iou2ds: list(B) 1 x T x T ## GT
            gt_proposals: list(B) num_of_labels x 2, with format [start, end], unit being seconds (frame/fps)
            separate_iou2ds: list(sum of all GT in batch)
            original_sent_feats: list(all sentences) x C
            num_sentences: num of samples in each sample [1, 2, 2, 3, 1, ...]
            intra_query_mask: num_multi_gt_samples x num_multi_samples np array mask (0 means query_template is the same, don't use as negative samples)
        """

        # prepare tensors
        B, C, _, _ = feat2ds.size() ## 16, 256
        feat1ds = feat2ds.masked_select(self.mask2d).reshape(B, C, -1) ## reshape the 2d map upper triangle to 1d sequence of feature [bs, C, 16*16/2+16/2]
        feat1ds_norm = F.normalize(feat1ds, dim=1)  # B x C x num_sparse_selected_proposal, ## normalize embedding feature
        
        sent_feat_cat = torch.cat(sent_feats, 0)  # sum(num_sent) x C, whole batch [bs, C]
        sum_num_sent = sent_feat_cat.size(0) ## one per sample, so same as bs
        sent_feat_cat_norm = F.normalize(sent_feat_cat, dim=1)  # sum(num_sent)=B x C, whole batch ## normalize embedding feature

        ## intra_query loss sentences feat for this batch
        original_sent_feats = torch.cat(original_sent_feats, dim=0)
        original_sent_feats = F.normalize(original_sent_feats, dim=1)
        

        mask_for_multi_gt = np.array([num > 1 for num in num_sentences])
        mask_for_multi_gt = np.repeat(mask_for_multi_gt, num_sentences)
        ## select samples that are multi-gt
        multi_gt_ori_sent_feats = [ori_sent_feat.unsqueeze(0) for ori_sent_feat, flag in zip(original_sent_feats, mask_for_multi_gt) if flag]
        multi_gt_samples_num_list = [num for num in num_sentences if num > 1]

        if mask_for_multi_gt.any(): ## if any one is true
            multi_gt_ori_sent_feats = torch.cat(multi_gt_ori_sent_feats, 0)


        ## for inter-modal vid loss
        vid_pos_list = []
        vid_neg_list = []
        
        ## for inter-modal sent loss
        sent_pos_list = []
        sent_neg_list = [] 
        
        ## for intra-modal vid loss
        intra_vid_pos_list = []  
        intra_vid_neg_list = []

        ## for intra-modal query loss
        intra_query_pos_list = []
        intra_query_neg_list = []


        num_gt_list = []
        label_count = 0
        multi_gt_sample_count = 0
        ## Iterate through a batch
        for i, (sent_feat, combined_iou2d, labels) in enumerate(zip(sent_feats, iou2ds, gt_proposals)):  # each video in the batch
            ## iou2d: [num_query, 16, 16] 
            num_gt = labels.shape[0]
            num_gt_list.append(num_gt)
            separate_iou2d = separate_iou2ds[label_count:label_count+num_gt]
            feat1d = feat1ds_norm[i, :, :]              # C x num_sparse_selected_proposal
            sent_feat = F.normalize(sent_feat, dim=1)   # num_sent=1 x C

            ## need to choose all multi-label index
            pos_feat_list = []
            pos_iou_list = []
            for gt_idx, label_iou2d in enumerate(separate_iou2d):
                iou1d = label_iou2d.masked_select(self.mask2d).reshape(sent_feat.size(0), -1)                           # num_sent=1 x num_sparse_selected_proposal
                ## record their IoU
                topk_iou, topk_index = torch.topk(iou1d, self.top_k, dim=-1)                                            # num_sent=1 x top_k (top_k=1)
                #print(f"topk_iou:{topk_iou.shape}, {topk_iou}") ## shape [1, top_k], [[0.9353, 0.8967, 0.8921]]
                selected_feat = feat1d.index_select(dim=1, index=topk_index.reshape(-1)).reshape(C, -1, self.top_k)     # C x num_sent=1 x top_k
                selected_feat = selected_feat.permute(1, 2, 0)  # num_sent=1 x top_k x C=256
                pos_feat_list.append(selected_feat)
                pos_iou_list.append(topk_iou.squeeze())

            selected_pos_feat = torch.cat(pos_feat_list, dim=0) ## num_gts x top_k x C
            pos_iou_feat = torch.cat(pos_iou_list, dim=0)
            #print(f"pos_iou_feat:{pos_iou_feat.shape}, {pos_iou_feat}")


            ## Video moments -> Querys
            #vid_pos = torch.matmul(selected_pos_feat, sent_feat.unsqueeze(2)).reshape(-1, self.top_k) - self.margin     ## (num_gt*top_k) x num_query=1            
            vid_pos = torch.matmul(selected_pos_feat, sent_feat.unsqueeze(2)).reshape(-1, self.top_k) 
            vid_neg = torch.mm(selected_pos_feat.view(-1, C), sent_feat_cat_norm.t()).reshape(-1, self.top_k, sum_num_sent)    # num_gt x topk=1 x sum(num_sent), mm of (num_sent*top_k x C) and (C x sum(num_sent))
            #print(f"vid pos:{vid_pos.shape}") ## num_gt*num_query=1 X top_k
            #print(f"vid neg:{vid_neg.shape}") ## num_gt, top_k=5, whole_batch_query 

            vid_pos_list.append(vid_pos) ## (num_gts*num_query=1, top_k), ex. (2, 5)
            vid_neg_list.append(vid_neg) ## (num_gts, top_k, whole_batch_query)   ex. (2, 5, 24)
            ##############################################################################################

            # Query -> video moments
            ## pos pairs
            sent_pos_list.append(vid_pos.clone())  ## pos are the same, (num_gt x top_k)

            ## neg pairs intra-sample
            sent_neg_same_video = torch.mm(sent_feat, feat1d)  # num_sent=1 x num_sparse_selected_proposal, query to all video momoents
            combined_iou1d = combined_iou2d.masked_select(self.mask2d).reshape(sent_feat.size(0), -1)  
            iou_neg_mask = (combined_iou1d < self.neg_iou).float()    # only keep the iou < 0.5 proposals as negative samples in the same video
            sent_neg_same_video = iou_neg_mask * sent_neg_same_video  # num_sent=1 x num_sparse_selected_proposal

            ## neg pairs cross-sample
            #pruning_index_list = [batch_idx for batch_idx, x in enumerate(query_mask) if not x]
            feat1d_other_video = feat1ds_norm.index_select(dim=0, index=torch.arange(B, device=feat2ds.device)[torch.arange(B, device=feat2ds.device) != i])     # (B-1) x C x num_sparse_selected_proposal
            #print(f"feat1d_other_video:{feat1d_other_video.shape}") ## (B-1) x C x 136

            feat1d_other_video = feat1d_other_video.transpose(1, 0).reshape(C, -1)  # C x ((B-1) x num_sparse_selected_proposal), 256 x ((bs-1)*136)            
            ## query and other sample video proposals do cosine similarity
            sent_neg_other_video = torch.mm(sent_feat, feat1d_other_video)      # num_sent=1 x ((B-1) x num_sparse_selected_proposal)
            
            sent_neg_all = [vid_pos.clone().unsqueeze(2),  ## num_gt*num_query=1 x top_k x 1
                            sent_neg_same_video.unsqueeze(1).repeat(num_gt, self.top_k, 1),   # num_gt x top_k x num_sparse_selected_proposal=136
                            sent_neg_other_video.unsqueeze(1).repeat(num_gt, self.top_k, 1)]  # num_gt x top_k x ((B-1) x num_sparse_selected_proposal)

            sent_neg_list.append(torch.cat(sent_neg_all, dim=2))  # list of  num_gt x topk x (1 + num_same + num_other)

            ##############################################################################################
            ## intra-modal vid loss
            ## selected_pos_feat: num_gts x top_k x C
            intra_vid_pos = torch.mm(selected_pos_feat.reshape(-1, C), selected_pos_feat.reshape(-1, C).t()) ## num_gt*top_k x num_gt*top_k
            #intra_vid_pos = torch.triu(intra_vid_pos, diagonal=1) ## keep value of upper triangle (not including diagonal line)
            #intra_mask = torch.ones(intra_vid_pos.shape[0], intra_vid_pos.shape[1]) - torch.eye(intra_vid_pos.shape[0])
            #print(f"intra_mask:{intra_mask.shape}, \n:{intra_mask}")
            intra_vid_pos = intra_vid_pos.reshape(-1, 1)   ## (num_gt*top_k)^2 x 1
            intra_vid_pos_list.append(intra_vid_pos)

            feat1d_neg_same_video = feat1d * iou_neg_mask ## neg vid moments, C * 136
            ## selected_pos_feat:  num_gts x top_k x C
            intra_vid_neg = torch.mm(selected_pos_feat.reshape(-1, C), feat1d_neg_same_video) ## num_gt*top_k x 136
            intra_vid_neg = intra_vid_neg.repeat_interleave(intra_vid_neg.shape[0], dim=0)  ## (num_gt*top_k)^2 x 136

            intra_vid_neg_all = [
                                    intra_vid_pos.unsqueeze(2),  ## (num_gt*top_1)^2 X 1 X 1
                                    intra_vid_neg.unsqueeze(1),  ## (num_gt*top_1)^2 X 1 X 136
                                ]
            intra_vid_neg_list.append(torch.cat(intra_vid_neg_all, dim=2)) ## (num_gt*top_1)^2 X 1 X (1 + 136)


            ##############################################################################################
            ## intra-modal query loss
            if num_gt > 1: ## only have intra_query loss in multi-gt samples
                ## Pos
                ## find the selected pos original queries
                intra_query_pos_feats = original_sent_feats[label_count:label_count+num_gt]
        
                intra_query_pos = torch.mm(intra_query_pos_feats, sent_feat.t()) ## [num_gt, C] x [C, 1] -> [num_gt, 1]
                intra_query_pos_list.append(intra_query_pos)
                
                ## Neg
                ## intra_query_mask from query_template level to original query level
                intra_query_neg_mask = np.repeat(intra_query_mask[multi_gt_sample_count], multi_gt_samples_num_list) 
                intra_query_neg_mask = torch.from_numpy(intra_query_neg_mask).unsqueeze(1).to(feat2ds.device)
                ## prune neg samples that has same query_template 
                intra_query_neg_queries = torch.mul(multi_gt_ori_sent_feats, intra_query_neg_mask)  ## [sum_num, C]
                intra_query_neg = torch.mm(sent_feat, multi_gt_ori_sent_feats.t()) ## [1, C] x [C, sum_num] -> [1, sum_num]
                intra_query_neg = intra_query_neg.repeat(num_gt, 1)
                intra_query_neg_all = [
                                            intra_query_pos.unsqueeze(2), ## [num_gt, 1, 1]
                                            intra_query_neg.unsqueeze(1), ## [num_gt, 1, sum_num]
                                      ]

                intra_query_neg_list.append(torch.cat(intra_query_neg_all, dim=2))  ## num_gt X 1 X (1 + sum_num)

                multi_gt_sample_count += 1 ## index of multi-gt sample in a batch


            label_count += num_gt
        
        #####################################################################################3
        ## Video moments -> Query
        vid_pos = torch.cat(vid_pos_list, dim=0) / self.T_v   # 31, top_k
        ## Video moments -> all query in the batch
        vid_neg = torch.cat(vid_neg_list, dim=0) / self.T_v  # 31, top_k, bs 
                
        ## positive pairs in denominator need to - m
        #margin_mask = torch.zeros(vid_neg.shape[-2:], device=feat2ds.device)
        #label_count = 0
        ## pos pairs in the denominator
        #for i, num_gt in enumerate(num_gt_list):
        #    margin_mask[label_count:label_count+num_gt, i] = 0.4
        #    label_count += num_gt
        #vid_neg = (vid_neg - margin_mask.unsqueeze(0)) / self.T_v   # top_k x this_cat_to_be_sum(num_sent) x sum(num_sent)=bs
        vid_neg_exp = torch.exp(vid_neg)  ## 1, 31, bs
        ## p(is|v) video moment -> many query loss 
        loss_vid = -(vid_pos - torch.log(vid_neg_exp.sum(dim=2, keepdim=False))).mean() ## vid_pos: 31, top_k

        ###############################################################################################################
        sent_pos = torch.cat(sent_pos_list, dim=0) / self.T_s ## 31, top_k
        sent_neg = torch.cat(sent_neg_list, dim=0) / self.T_s ## 31, top_k, 2177
        sent_neg_exp = torch.exp(sent_neg) 
        ## p(iv|s) query -> many video moments loss
        loss_sent = -(sent_pos - torch.log(sent_neg_exp.sum(dim=2, keepdim=False) + self.eps)).mean() ## eps=1e-6

        ###############################################################################################################
        intra_vid_pos = torch.cat(intra_vid_pos_list, dim=0) / self.T_v
        intra_vid_neg = torch.cat(intra_vid_neg_list, dim=0) / self.T_v
        #print(f"intra_vid pos:{intra_vid_pos.shape}, neg:{intra_vid_neg.shape}")
        intra_vid_neg_exp = torch.exp(intra_vid_neg)
        loss_intra_vid = -(intra_vid_pos - torch.log(intra_vid_neg_exp.sum(dim=2, keepdim=False))).mean()
        
        ###############################################################################################################
        ## not always have multi-gt samples
        if len(intra_query_pos_list) > 0:
            intra_query_pos = torch.cat(intra_query_pos_list, dim=0) / self.T_s
            intra_query_neg = torch.cat(intra_query_neg_list, dim=0) / self.T_s
            #print(f"intra_query pos:{intra_query_pos.shape}, neg:{intra_query_neg.shape}")
            intra_query_neg_exp = torch.exp(intra_query_neg)
            loss_intra_query = -(intra_query_pos - torch.log(intra_query_neg_exp.sum(dim=2, keepdim=False))).mean()
        else:
            ## no multi-gt samples
            loss_intra_query = torch.zeros(1,)[0].to(feat2ds.device)


        return loss_vid, loss_sent, loss_intra_vid, loss_intra_query


def build_contrastive_loss_multi_labels(cfg, mask2d):
    return ContrastiveLoss_multi_labels(cfg, mask2d)