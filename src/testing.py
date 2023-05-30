import os
import csv
import json
from zipfile import ZipFile
from os.path import basename

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.multiprocessing
from torch.nn import SyncBatchNorm
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

import src.dist as dist
from src.evaluation import calculate_recall, calculate_multi_recall, calculate_mAPs, calculate_mAPs_no_mean, recall_name
from src.misc import print_metrics, print_recall, print_mAPs, print_multi_recall, construct_class
from src.models.main import MMN
from src.utils import nms, scores2ds_to_moments, moments_to_iou2ds, iou2ds_to_iou2d, plot_moments_on_iou2d


def qv_testing_loop(config):
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

    # Val
    pred_submission = []
    pred_moments = []
    true_moments = []
    pbar = tqdm(val_loader, ncols=0, leave=False, desc="Inferencing val set")
    for batch, batch_info in pbar:
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.no_grad():
            *_, scores2d, mask2d = model(**batch)

        out_moments, out_scores1ds = scores2ds_to_moments(scores2d, mask2d)
        pred_moments_batch = nms(out_moments, out_scores1ds, config.nms_threshold)
        pred_moments_batch = dist.gather_dict(pred_moments_batch, to_cpu=True)
        pred_moments.append(pred_moments_batch)

        true_moments_batch = {
            'tgt_moments': batch['tgt_moments'],
            'num_targets': batch['num_targets'],
        }
        true_moments_batch = dist.gather_dict(true_moments_batch, to_cpu=True)
        true_moments.append(true_moments_batch)

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
                pred_submission.append(json.dumps({
                    'qid': qid.item(),
                    'query': query,
                    'vid': vid,
                    'pred_relevant_windows': pred_relevant_windows.tolist(),
                }))
                shift += num_proposal

    with open(os.path.join(config.logdir, 'hl_val_submission.jsonl'), 'w') as f:
        for line in pred_submission:
            f.write(line + '\n')

    recall = calculate_recall(
        pred_moments, true_moments, config.recall_Ns, config.recall_IoUs)
    mAPs = calculate_mAPs(pred_moments, true_moments)

    print_metrics(mAPs, recall)

    # test
    pred_submission = []
    pred_moments = []
    true_moments = []
    pbar = tqdm(test_loader, ncols=0, leave=False, desc="Inferencing test set")
    for batch, batch_info in pbar:
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.no_grad():
            *_, scores2d, mask2d = model(**batch)

        out_moments, out_scores1ds = scores2ds_to_moments(scores2d, mask2d)
        pred_moments_batch = nms(out_moments, out_scores1ds, config.nms_threshold)
        pred_moments_batch = dist.gather_dict(pred_moments_batch, to_cpu=True)
        pred_moments.append(pred_moments_batch)

        true_moments_batch = {
            'tgt_moments': batch['tgt_moments'],
            'num_targets': batch['num_targets'],
        }
        true_moments_batch = dist.gather_dict(true_moments_batch, to_cpu=True)
        true_moments.append(true_moments_batch)

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
                pred_submission.append(json.dumps({
                    'qid': qid.item(),
                    'query': query,
                    'vid': vid,
                    'pred_relevant_windows': pred_relevant_windows.tolist(),
                }))
                shift += num_proposal

    with open(os.path.join(config.logdir, 'hl_test_submission.jsonl'), 'w') as f:
        for line in pred_submission:
            f.write(line + '\n')

    with ZipFile(os.path.join(config.logdir, 'submission.zip'), 'w') as zip_obj:
        val_submission_path = os.path.join(config.logdir, 'hl_val_submission.jsonl')
        test_submission_path = os.path.join(config.logdir, 'hl_test_submission.jsonl')
        zip_obj.write(val_submission_path, basename(val_submission_path))
        zip_obj.write(test_submission_path, basename(test_submission_path))


def testing_loop(config):
    device = dist.get_device()
    # test Dataset and DataLoader
    test_dataset = construct_class(config.TestDataset)
    test_sampler = DistributedSampler(test_dataset, shuffle=False, seed=config.seed)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.test_batch_size // dist.get_world_size(),
        collate_fn=test_dataset.collate_fn,
        sampler=test_sampler,
        num_workers=min(torch.get_num_threads(), 8),
    )
    # Multi-test Dataset and DataLoader
    if "charades" in config.TestDataset or "activity" in config.TestDataset:
        # re-annotated multi-test Dataset and DataLoader
        multi_test_dataset = construct_class(config.MultiTestDataset)
        multi_test_sampler = DistributedSampler(multi_test_dataset,
                                                shuffle=False,
                                                seed=config.seed)
        multi_test_loader = DataLoader(
            dataset=multi_test_dataset,
            batch_size=config.test_batch_size // dist.get_world_size(),
            collate_fn=multi_test_dataset.collate_fn,
            sampler=multi_test_sampler,
            num_workers=min(torch.get_num_threads(), 8),
        )

    # model
    model_local = MMN(
        backbone=config.backbone,
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
        resnet=config.resnet,
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

    if "qv" in config.TestDataset:
        pred_moments = []
        true_moments = []
        nms_thres = np.arange(0.1, 1, 0.1)   # [0.2, ..., 0.9]
        nms_thres = [round(nms_t, 2) for nms_t in nms_thres]
        pred_moments_nms_list = [[] for _ in range(len(nms_thres))]
        sample_id = 0
        plot_dir = os.path.join(config.logdir, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        for batch, batch_info in tqdm(test_loader, ncols=0, leave=False, desc="Inferencing"):
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.no_grad():
                *_, scores2ds, mask2d = model(**batch)

            true_moments_batch = {
                'tgt_moments': batch['tgt_moments'],
                'num_targets': batch['num_targets'],    # [S]
            }
            true_moments_batch = dist.gather_dict(true_moments_batch, to_cpu=True)
            true_moments.append(true_moments_batch)

            out_moments, out_scores1ds = scores2ds_to_moments(scores2ds, mask2d)
            pred_moments_batch = nms(out_moments, out_scores1ds, config.nms_threshold)
            pred_moments_batch = dist.gather_dict(pred_moments_batch, to_cpu=True)
            pred_moments.append(pred_moments_batch)

            # ploting batch
            iou2ds = moments_to_iou2ds(batch['tgt_moments'], config.num_clips)          # [M, N, N]
            iou2d = iou2ds_to_iou2d(iou2ds, batch['num_targets'])                       # [S, N, N], separate to combined
            iou2d = iou2d.detach().cpu()
            shift = 0
            for batch_idx, scores2d in enumerate(scores2ds.cpu()):
                # nms(pred)
                num_proposals = pred_moments_batch['num_proposals'][batch_idx]
                nms_moments = pred_moments_batch["out_moments"][shift: shift + num_proposals]  # [num_proposals, 2]
                nms_moments = (nms_moments * config.num_clips).round().long()
                plot_path = os.path.join(
                    plot_dir,
                    f"{sample_id:05d}.jpg"
                )
                if sample_id % 50 == 0:
                    plot_moments_on_iou2d(
                        iou2d[batch_idx], scores2d, nms_moments, plot_path
                    )
                shift = shift + num_proposals
                sample_id += 1

            for idx, nms_t in enumerate(nms_thres):
                pred_moments_batch = nms(out_moments, out_scores1ds, nms_t)
                pred_moments_batch = dist.gather_dict(pred_moments_batch, to_cpu=True)  # dict
                pred_moments_nms_list[idx].append(pred_moments_batch)  # list of dict

        # original mAP_avg
        sample_mAPs, _ = calculate_mAPs_no_mean(pred_moments, true_moments, max_proposals=10)
        print(f"Original mAP_avg:{round(sample_mAPs.mean().item() * 100, 2)}")

        sample_mAPs_list = []
        for idx, pred_moments in enumerate(pred_moments_nms_list):
            sample_mAPs, num_targets = calculate_mAPs_no_mean(pred_moments, true_moments, max_proposals=10)
            sample_mAPs_list.append(sample_mAPs)
        sample_mAPs_to_nms_t = torch.stack(sample_mAPs_list, dim=1)  # [1550, num_nms_t]

        # filter some mAP results
        mask = torch.ones(sample_mAPs_to_nms_t.shape[0])
        for idx, mAPs_to_nms_t in enumerate(sample_mAPs_to_nms_t):
            if all(element == mAPs_to_nms_t[0] for element in mAPs_to_nms_t):
                mask[idx] = 0
            # if (max(mAPs_to_nms_t) - min(mAPs_to_nms_t)) <= 0.01:
            #     mask[idx] = 0

        # save sample_mAPs_to_nms_t as csv
        with open('nms.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([''] + nms_thres + ['mask'])
            for idx, mAPs_to_nms_t in enumerate(sample_mAPs_to_nms_t):
                writer.writerow([f"{idx}"] + mAPs_to_nms_t.tolist() + [mask[idx].item()])

        # ideal mAP results
        max_mAPs, max_nms_idxs = torch.max(sample_mAPs_to_nms_t, dim=1)
        ideal_nms_t_result = sample_mAPs_to_nms_t[range(sample_mAPs_to_nms_t.shape[0]), max_nms_idxs.long()]
        print(f"ideal_nms_t_result:{round(ideal_nms_t_result.mean().item() * 100, 2)}")

        sample_mAPs_to_nms_t = sample_mAPs_to_nms_t[mask.bool()]
        print(f"filtered sample_mAPs:{sample_mAPs_to_nms_t.shape}")
        max_mAPs, max_nms_idxs = torch.max(sample_mAPs_to_nms_t, dim=1)

        plot_value = torch.zeros(max(num_targets), len(nms_thres))   # ex. [25, 8]
        for idx, (num_target, max_mAP) in enumerate(zip(num_targets, max_mAPs)):
            for element_idx, element in enumerate(sample_mAPs_to_nms_t[idx]):
                if element == max_mAP:
                    plot_value[num_target - 1, element_idx] += 1

        with open('nms_count.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([''] + nms_thres)
            multi_sum = torch.zeros_like(plot_value[0])
            for idx, best_nms_count in enumerate(plot_value):
                num_target = idx + 1
                writer.writerow([f'{num_target}'] + best_nms_count.tolist())
                if idx == 0:
                    single = best_nms_count
                else:
                    multi_sum += best_nms_count
            writer.writerow(['Single_target'] + single.tolist())
            writer.writerow(['Multi_target'] + multi_sum.tolist())

    elif "charades" in config.TestDataset:
        # Testing set
        pred_moments = []
        true_moments = []
        for batch, batch_info in tqdm(test_loader, ncols=0, leave=False, desc="Inferencing Testing set"):
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.no_grad():
                *_, scores2d, mask2d = model(**batch)

            out_moments, out_scores1ds = scores2ds_to_moments(scores2d, mask2d)
            pred_moments_batch = nms(out_moments, out_scores1ds, config.nms_threshold)
            pred_moments_batch = dist.gather_dict(pred_moments_batch, to_cpu=True)
            pred_moments.append(pred_moments_batch)

            true_moments_batch = {
                'tgt_moments': batch['tgt_moments'],
                'num_targets': batch['num_targets'],    # [S]
            }
            true_moments_batch = dist.gather_dict(true_moments_batch, to_cpu=True)
            true_moments.append(true_moments_batch)

        recall = calculate_recall(
            pred_moments, true_moments,
            config.recall_Ns, config.recall_IoUs
        )
        print(f"Testing set")
        print_recall(recall)

        # Multi-Testing set
        pred_moments = []
        true_moments = []
        for batch, batch_info in tqdm(multi_test_loader, ncols=0, leave=False, desc="Inferencing Multi-test set"):
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.no_grad():
                *_, scores2d, mask2d = model(**batch)

            out_moments, out_scores1ds = scores2ds_to_moments(scores2d, mask2d)
            pred_moments_batch = nms(out_moments, out_scores1ds, config.nms_threshold)
            pred_moments_batch = dist.gather_dict(pred_moments_batch, to_cpu=True)
            pred_moments.append(pred_moments_batch)

            true_moments_batch = {
                'tgt_moments': batch['tgt_moments'],
                'num_targets': batch['num_targets'],
            }
            true_moments_batch = dist.gather_dict(true_moments_batch, to_cpu=True)
            true_moments.append(true_moments_batch)

        multi_test_recall = calculate_multi_recall(
            pred_moments, true_moments,
            [5,], config.recall_IoUs
        )
        print(f"Multi test set")
        print_multi_recall(multi_test_recall)

    elif "activity" in config.TestDataset:
        # Testing set
        pred_moments = []
        true_moments = []
        for batch, batch_info in tqdm(test_loader, ncols=0, leave=False, desc="Inferencing Testing set"):
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.no_grad():
                *_, scores2d, mask2d = model(**batch)

            out_moments, out_scores1ds = scores2ds_to_moments(scores2d, mask2d)
            pred_moments_batch = nms(out_moments, out_scores1ds, config.nms_threshold)
            pred_moments_batch = dist.gather_dict(pred_moments_batch, to_cpu=True)
            pred_moments.append(pred_moments_batch)

            true_moments_batch = {
                'tgt_moments': batch['tgt_moments'],
                'num_targets': batch['num_targets'],
            }
            true_moments_batch = dist.gather_dict(true_moments_batch, to_cpu=True)
            true_moments.append(true_moments_batch)

        recall = calculate_recall(
            pred_moments, true_moments,
            config.recall_Ns, config.recall_IoUs
        )
        print(f"Testing set")
        print_recall(recall)

        # Multi-Testing set
        pred_moments = []
        true_moments = []
        for batch, batch_info in tqdm(multi_test_loader, ncols=0, leave=False, desc="Inferencing Multi-test set"):
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.no_grad():
                *_, scores2d, mask2d = model(**batch)

            out_moments, out_scores1ds = scores2ds_to_moments(scores2d, mask2d)
            pred_moments_batch = nms(out_moments, out_scores1ds, config.nms_threshold)
            pred_moments_batch = dist.gather_dict(pred_moments_batch, to_cpu=True)
            pred_moments.append(pred_moments_batch)

            true_moments_batch = {
                'tgt_moments': batch['tgt_moments'],
                'num_targets': batch['num_targets'],
            }
            true_moments_batch = dist.gather_dict(true_moments_batch, to_cpu=True)
            true_moments.append(true_moments_batch)

        multi_test_recall = calculate_multi_recall(
            pred_moments, true_moments,
            [5,], config.recall_IoUs
        )
        print(f"Multi test set")
        print_multi_recall(multi_test_recall)
