import os
import json
from zipfile import ZipFile
from os.path import basename

import torch
import torch.multiprocessing
from torch.nn import SyncBatchNorm
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

import src.dist as dist
from src.evaluation import calculate_recall, calculate_mAPs, recall_name
from src.misc import print_table, construct_class
from src.models.model import MMN, MMN_bbox_reg
from src.utils import (
    nms, scores2ds_to_moments, 
    moments_to_iou2ds, moments_to_rescaled_iou2ds, iou2ds_to_iou2d
)


def testing_loop(config):
    device = dist.get_device()

    # val Dataset and DataLoader
    val_dataset = construct_class("src.datasets.qvhighlights.QVHighlightsVal2s") 
    val_sampler = DistributedSampler(val_dataset, shuffle=False, seed=config.seed)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.test_batch_size // dist.get_world_size(),
        collate_fn=val_dataset.collate_fn,
        sampler=val_sampler,
        num_workers=min(torch.get_num_threads(), 8),
    )

    # test Dataset and DataLoader
    test_dataset = construct_class("src.datasets.qvhighlights.QVHighlightsTest2s") 
    test_sampler = DistributedSampler(test_dataset, shuffle=False, seed=config.seed)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.test_batch_size // dist.get_world_size(),
        collate_fn=test_dataset.collate_fn,
        sampler=test_sampler,
        num_workers=min(torch.get_num_threads(), 8),
    )

    # model
    model_local = MMN(
        num_init_clips=config.num_init_clips,
        feat1d_in_channel=test_dataset.get_feat_dim(),
        feat1d_out_channel=config.feat1d_out_channel,
        feat1d_pool_kernel_size=config.feat1d_pool_kernel_size,
        feat1d_pool_stride_size=config.num_init_clips // config.num_clips,
        feat2d_pool_counts=config.feat2d_pool_counts,
        conv2d_hidden_channel=config.conv2d_hidden_channel,
        conv2d_kernel_size=config.conv2d_kernel_size,
        conv2d_num_layers=config.conv2d_num_layers,
        joint_space_size=config.joint_space_size,
        dual_space=config.dual_space,
    ).to(device)
    # load from checkpoint
    ckpt = torch.load(os.path.join(config.logdir, 'best.pth'))
    model_local.load_state_dict(ckpt['model'])
    # DDP
    model = SyncBatchNorm.convert_sync_batchnorm(model_local)
    model = DistributedDataParallel(
        model, device_ids=[device], find_unused_parameters=True)
    model.eval()
    
    ## mAP metric groups
    mAP_keys_group_1 = ["avg_mAP", "mAP@0.50", "mAP@0.75", 
                        "single_avg_mAP", "single_mAP@0.50", "single_mAP@0.75",
                        "multi_avg_mAP", "multi_mAP@0.50", "multi_mAP@0.75",]
    mAP_keys_group_2 = ["short_avg_mAP", "short_mAP@0.50", "short_mAP@0.75",
                        "medium_avg_mAP", "medium_mAP@0.50", "medium_mAP@0.75",
                        "long_avg_mAP", "long_mAP@0.50", "long_mAP@0.75"]


    ## Val
    pred_val_submission = []
    pred_moments = []
    true_moments = []
    pbar = tqdm(val_loader, ncols=0, leave=False, desc="Inferencing val set")
    for batch, batch_info in pbar:
        batch = {key: value.to(device) for key, value in batch.items()}
        true_moments_batch = {
            'tgt_moments': batch['tgt_moments'],
            'num_targets': batch['num_targets'],
        }
        true_moments_batch = dist.gather_dict(true_moments_batch, to_cpu=True)
        true_moments.append(true_moments_batch)
        
        with torch.no_grad():
            *_, scores2ds, mask2d = model(**batch)

        out_moments, out_scores1ds = scores2ds_to_moments(scores2ds, mask2d)
        pred_moments_batch = nms(out_moments, out_scores1ds, config.nms_threshold)
        pred_moments_batch = dist.gather_dict(pred_moments_batch, to_cpu=True)
        pred_moments.append(pred_moments_batch)

        assert len(batch_info['qids']) == len(batch_info['sentences']) == len(batch_info['vid']) == len(batch_info['duration'])
        shift = 0
        num_proposals = iter(pred_moments_batch['num_proposals'])
        for qids, sentences, vid, duration, in zip(
            batch_info['qids'],
            batch_info['sentences'],
            batch_info['vid'],
            batch_info['duration']
        ):
            assert len(qids) == len(sentences)
            for qid, query in zip(qids, sentences):
                num_proposal = next(num_proposals)
                out_scores1d = pred_moments_batch['out_scores1ds'][shift: shift + num_proposal]
                out_moments = pred_moments_batch['out_moments'][shift: shift + num_proposal] * duration
                pred_relevant_windows = torch.cat([out_moments, out_scores1d.unsqueeze(-1)], dim=-1)
                pred_val_submission.append(json.dumps({
                    'qid': qid.item(),
                    'query': query,
                    'vid': vid,
                    'pred_relevant_windows': pred_relevant_windows.tolist(),
                }))
                shift += num_proposal

    recall = calculate_recall(
        pred_moments, true_moments, config.recall_Ns, config.recall_IoUs)
    mAPs = calculate_mAPs(pred_moments, true_moments)
    ## split mAPs table, too long to print on terminal
    mAPs_group_1 = {key: mAPs[key] for key in mAP_keys_group_1}
    mAPs_group_2 = {key: mAPs[key] for key in mAP_keys_group_2}
    print_table(epoch=0, rows={'val': recall})
    print_table(epoch=0, rows={'val': mAPs_group_1})
    print_table(epoch=0, rows={'val': mAPs_group_2})

    '''
    ## test
    pred_test_submission = []
    pred_moments = []
    pbar = tqdm(test_loader, ncols=0, leave=False, desc="Inferencing test set")
    for batch, batch_info in pbar:
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.no_grad():
            *_, scores2ds, mask2d = model(**batch)

        out_moments, out_scores1ds = scores2ds_to_moments(scores2ds, mask2d)
        pred_moments_batch = nms(out_moments, out_scores1ds, config.nms_threshold)
        pred_moments_batch = dist.gather_dict(pred_moments_batch, to_cpu=True)
        pred_moments.append(pred_moments_batch)

        assert len(batch_info['qids']) == len(batch_info['sentences']) == len(batch_info['vid']) == len(batch_info['duration'])
        shift = 0
        num_proposals = iter(pred_moments_batch['num_proposals'])
        for qids, sentences, vid, duration, in zip(
            batch_info['qids'],
            batch_info['sentences'],
            batch_info['vid'],
            batch_info['duration']
        ):
            assert len(qids) == len(sentences)
            for qid, query in zip(qids, sentences):
                num_proposal = next(num_proposals)
                out_scores1d = pred_moments_batch['out_scores1ds'][shift: shift + num_proposal]
                out_moments = pred_moments_batch['out_moments'][shift: shift + num_proposal] * duration
                pred_relevant_windows = torch.cat([out_moments, out_scores1d.unsqueeze(-1)], dim=-1)
                pred_test_submission.append(json.dumps({
                    'qid': qid.item(),
                    'query': query,
                    'vid': vid,
                    'pred_relevant_windows': pred_relevant_windows.tolist(),
                }))
                shift += num_proposal


    ## Write submission file to zip file
    with open(os.path.join(config.logdir, 'hl_val_submission.jsonl'), 'w') as f:
        for line in pred_val_submission:
            f.write(line + '\n')
    with open(os.path.join(config.logdir, 'hl_test_submission.jsonl'), 'w') as f:
        for line in pred_test_submission:
            f.write(line + '\n')
    with ZipFile(os.path.join(config.logdir, 'submission.zip'), 'w') as zip_obj:
        val_submission_path = os.path.join(config.logdir, 'hl_val_submission.jsonl')
        test_submission_path = os.path.join(config.logdir, 'hl_test_submission.jsonl')
        zip_obj.write(val_submission_path, basename(val_submission_path))
        zip_obj.write(test_submission_path, basename(test_submission_path))
    '''

def testing_loop_bbox_reg(config):
    device = dist.get_device()

    # val Dataset and DataLoader
    val_dataset = construct_class("src.datasets.qvhighlights.QVHighlightsVal2s") 
    val_sampler = DistributedSampler(val_dataset, shuffle=False, seed=config.seed)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.test_batch_size // dist.get_world_size(),
        collate_fn=val_dataset.collate_fn,
        sampler=val_sampler,
        num_workers=min(torch.get_num_threads(), 8),
    )

    # test Dataset and DataLoader
    test_dataset = construct_class("src.datasets.qvhighlights.QVHighlightsTest2s") 
    test_sampler = DistributedSampler(test_dataset, shuffle=False, seed=config.seed)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.test_batch_size // dist.get_world_size(),
        collate_fn=test_dataset.collate_fn,
        sampler=test_sampler,
        num_workers=min(torch.get_num_threads(), 8),
    )

    # model
    model_local = MMN_bbox_reg(
        num_init_clips=config.num_init_clips,
        feat1d_in_channel=test_dataset.get_feat_dim(),
        feat1d_out_channel=config.feat1d_out_channel,
        feat1d_pool_kernel_size=config.feat1d_pool_kernel_size,
        feat1d_pool_stride_size=config.num_init_clips // config.num_clips,
        feat2d_pool_counts=config.feat2d_pool_counts,
        conv2d_hidden_channel=config.conv2d_hidden_channel,
        conv2d_kernel_size=config.conv2d_kernel_size,
        conv2d_num_layers=config.conv2d_num_layers,
        joint_space_size=config.joint_space_size,
        dual_space=config.dual_space,
    ).to(device)
    # load from checkpoint
    ckpt = torch.load(os.path.join(config.logdir, 'best.pth'))
    model_local.load_state_dict(ckpt['model'])
    # DDP
    model = SyncBatchNorm.convert_sync_batchnorm(model_local)
    model = DistributedDataParallel(
        model, device_ids=[device], find_unused_parameters=True)
    model.eval()

    ## mAP metric groups
    mAP_keys_group_1 = ["avg_mAP", "mAP@0.50", "mAP@0.75", 
                        "single_avg_mAP", "single_mAP@0.50", "single_mAP@0.75",
                        "multi_avg_mAP", "multi_mAP@0.50", "multi_mAP@0.75",]
    mAP_keys_group_2 = ["short_avg_mAP", "short_mAP@0.50", "short_mAP@0.75",
                        "medium_avg_mAP", "medium_mAP@0.50", "medium_mAP@0.75",
                        "long_avg_mAP", "long_mAP@0.50", "long_mAP@0.75"]

    ## Val
    pred_val_submission = []
    pred_moments = []
    true_moments = []
    pbar = tqdm(val_loader, ncols=0, leave=False, desc="Inferencing val set")
    for batch, batch_info in pbar:
        batch = {key: value.to(device) for key, value in batch.items()}

        ## GT
        true_moments_batch = {
            'tgt_moments': batch['tgt_moments'],
            'num_targets': batch['num_targets'],
        }
        true_moments_batch = dist.gather_dict(true_moments_batch, to_cpu=True)
        true_moments.append(true_moments_batch)

        with torch.no_grad():
            *_, scores2ds, mask2d, bbox, scores1ds = model(**batch)
        
        #### testing: add correct offset to small targets' top-1 proposal and check the short mAP ceiling
        ## find top-1 proposal mask of small targets
        iou2ds = moments_to_rescaled_iou2ds(batch['tgt_moments'], config.num_clips)
        #iou2d = iou2ds_to_iou2d(iou2ds, batch['num_targets']) # [S, N, N] for each sentence
        S, _, _ = scores2ds.shape
        M = iou2ds.shape[0]
        
        ## select topk for each targets
        iou1ds = iou2ds.masked_select(mask2d).view(M, -1)       # [M, P]
        P = iou1ds.shape[-1]                                    # 1104                
        top_k = 3
        topk_idxs = iou1ds.topk(top_k, dim=1)[1]                # [M, top_k]

        num_targets = batch['num_targets']                      # [S]
        tgt_moments = batch['tgt_moments']                      # [M, 2]
        tgt_lens = tgt_moments[:, 1] - tgt_moments[:, 0]        # [M]
        small_clip_mask = tgt_lens.lt(0.05)                     # [M]
        small_target_mask = torch.zeros(S, P, device=device)    # [S, P], backgrounds are 0
        target_count = 0
        for sample_idx, num_target in enumerate(num_targets):
            for i in range(num_target):
                if small_clip_mask[target_count + i]:
                    for k in range(top_k):
                        topk_prop_idx = int(topk_idxs[target_count + i, k])
                        small_target_mask[sample_idx, topk_prop_idx] = 1
                        ## set this proposal to GT target moment
                        bbox[sample_idx, topk_prop_idx] = tgt_moments[target_count + i]
            target_count += num_target
             
        pred_moments_batch = nms(bbox, scores1ds, config.nms_threshold)
        pred_moments_batch = dist.gather_dict(pred_moments_batch, to_cpu=True)
        pred_moments.append(pred_moments_batch)

        ## plot result
        ##
                
        ## generate submission file
        assert len(batch_info['qids']) == len(batch_info['sentences']) == len(batch_info['vid']) == len(batch_info['duration'])
        shift = 0
        num_proposals = iter(pred_moments_batch['num_proposals'])
        for qids, sentences, vid, duration, in zip(
            batch_info['qids'],
            batch_info['sentences'],
            batch_info['vid'],
            batch_info['duration']
        ):
            assert len(qids) == len(sentences)
            for qid, query in zip(qids, sentences):
                num_proposal = next(num_proposals)
                out_scores1d = pred_moments_batch['out_scores1ds'][shift: shift + num_proposal]
                out_moments = pred_moments_batch['out_moments'][shift: shift + num_proposal] * duration
                pred_relevant_windows = torch.cat([out_moments, out_scores1d.unsqueeze(-1)], dim=-1)
                pred_val_submission.append(json.dumps({
                    'qid': qid.item(),
                    'query': query,
                    'vid': vid,
                    'pred_relevant_windows': pred_relevant_windows.tolist(),
                }))
                shift += num_proposal

    recall = calculate_recall(
        pred_moments, true_moments, config.recall_Ns, config.recall_IoUs)
    mAPs = calculate_mAPs(pred_moments, true_moments)
    ## split mAPs table, too long to print on terminal
    mAPs_group_1 = {key: mAPs[key] for key in mAP_keys_group_1}
    mAPs_group_2 = {key: mAPs[key] for key in mAP_keys_group_2}

    print_table(epoch=0, rows={'val': recall})
    print_table(epoch=0, rows={'val': mAPs_group_1})
    print_table(epoch=0, rows={'val': mAPs_group_2})

    '''
    ## test
    pred_test_submission = []
    pred_moments = []
    pbar = tqdm(test_loader, ncols=0, leave=False, desc="Inferencing test set")
    for batch, batch_info in pbar:
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.no_grad():
            *_, scores2ds, mask2d, bbox, scores1ds = model(**batch)
        
        pred_moments_batch = nms(bbox, scores1ds, config.nms_threshold)
        pred_moments_batch = dist.gather_dict(pred_moments_batch, to_cpu=True)
        pred_moments.append(pred_moments_batch)        

        ## plot result
        ##
        
        ## generate submission file
        assert len(batch_info['qids']) == len(batch_info['sentences']) == len(batch_info['vid']) == len(batch_info['duration'])
        shift = 0
        num_proposals = iter(pred_moments_batch['num_proposals'])
        for qids, sentences, vid, duration, in zip(
            batch_info['qids'],
            batch_info['sentences'],
            batch_info['vid'],
            batch_info['duration']
        ):
            assert len(qids) == len(sentences)
            for qid, query in zip(qids, sentences):
                num_proposal = next(num_proposals)
                out_scores1d = pred_moments_batch['out_scores1ds'][shift: shift + num_proposal]
                out_moments = pred_moments_batch['out_moments'][shift: shift + num_proposal] * duration
                pred_relevant_windows = torch.cat([out_moments, out_scores1d.unsqueeze(-1)], dim=-1)
                pred_test_submission.append(json.dumps({
                    'qid': qid.item(),
                    'query': query,
                    'vid': vid,
                    'pred_relevant_windows': pred_relevant_windows.tolist(),
                }))
                shift += num_proposal

    ## Write submission file to zip file
    with open(os.path.join(config.logdir, 'hl_val_submission.jsonl'), 'w') as f:
        for line in pred_val_submission:
            f.write(line + '\n')
    with open(os.path.join(config.logdir, 'hl_test_submission.jsonl'), 'w') as f:
        for line in pred_test_submission:
            f.write(line + '\n')
    with ZipFile(os.path.join(config.logdir, 'submission.zip'), 'w') as zip_obj:
        val_submission_path = os.path.join(config.logdir, 'hl_val_submission.jsonl')
        test_submission_path = os.path.join(config.logdir, 'hl_test_submission.jsonl')
        zip_obj.write(val_submission_path, basename(val_submission_path))
        zip_obj.write(test_submission_path, basename(test_submission_path))
    '''


