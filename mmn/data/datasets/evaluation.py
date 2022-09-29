from terminaltables import AsciiTable
from tqdm import tqdm
import logging
import torch
from mmn.data.datasets.utils import iou, score2d_to_moments_scores
from mmn.utils.comm import is_main_process
import os
import json

## for hungarian matching
from scipy.optimize import linear_sum_assignment
import numpy as np

from sklearn.cluster import DBSCAN

## for plotting
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def nms(moments, scores, thresh):
    scores, ranks = scores.sort(descending=True) ## high -> low, ranks: [136]
    moments = moments[ranks] ## [136, 2] from high to low
    suppressed = torch.clone(ranks).zero_().bool() ## all false initially, [136]
    numel = suppressed.numel() ## .numel() number of elements, 136
    for i in range(numel - 1): ## iterate through all element 0 -> 135
        if suppressed[i]:
            continue
        mask = iou(moments[i+1:], moments[i]) > thresh ## threshold = 0.5
        suppressed[i+1:][mask] = True

    remain_indices = (~suppressed).nonzero().squeeze()
    #print(f"ranks\n{ranks}")
    #print(f"remain_indices\n{(~suppressed).nonzero().squeeze()}")
    #print(f"after\n{ranks[remain_indices]}")

    return moments[~suppressed], ranks[remain_indices] ## ~suppressed complement, return not suppressed part


def evaluate(cfg, dataset, predictions, nms_thresh, recall_metrics=(1, 5)):
    ## predictions: {video_idx: {'contrastive': result1, 'iou': result2}
    """evaluate dataset using different methods based on dataset type.
    Args:
    Returns:
    """
    if not is_main_process():
        return
    if cfg.DATASETS.NAME == "tacos":
        iou_metrics = (0.1, 0.3, 0.5)
    elif cfg.DATASETS.NAME == "activitynet":
        iou_metrics = (0.3, 0.5, 0.7)
    elif cfg.DATASETS.NAME == "charades":
        iou_metrics = (0.5, 0.7)
    elif cfg.DATASETS.NAME == "charades_combined":
        iou_metrics = (0.5, 0.7)
    else:
        raise NotImplementedError("No support for %s dataset!" % cfg.DATASETS.NAME)
    dataset_name = dataset.__class__.__name__
    logger = logging.getLogger("mmn.inference")
    logger.info("Performing {} evaluation (Size: {}).".format(dataset_name, len(dataset)))

    num_recall_metrics, num_iou_metrics = len(recall_metrics), len(iou_metrics)
    recall_metrics = torch.tensor(recall_metrics) ## (1, 5)
    iou_metrics = torch.tensor(iou_metrics) ## (0.3, 0.5, 0.7)
    #print(f"shape:{predictions[0]['iou'].shape}") ## [num_query, 64, 64]
    num_clips = predictions[0]['iou'].shape[-1] ## 64
    table = [['R@{},IoU@{:.01f}'.format(i, torch.round(j*100)/100) for i in recall_metrics for j in iou_metrics]]
    recall_x_iou = torch.zeros(num_recall_metrics, num_iou_metrics) ## [2, 3] for activitynet

    num_instance = 0 ## number of video-query pairs
    for idx, result2d in tqdm(enumerate(predictions)):   # each video
        ## result2d['contrastive'] * 0.5 + 0.5: scale from [-1, 1] -> [0, 1]
        ## torch.pow(s_mm, 0.5): square root
        ## score_2d = s_mm^0.5 * s_iou
        score2d = torch.pow(result2d['contrastive'] * 0.5 + 0.5, cfg.TEST.CONTRASTIVE_SCORE_POW) * result2d['iou']
        #print(f"score_2d:{score2d.shape}") ## [num_query, 64, 64]


        duration = dataset.get_duration(idx) ## video length
        gt_moments = dataset.get_moment(idx) ## [[start1, end1], [start2, end2]...]
        for gt_moment, pred_score2d in zip(gt_moments, score2d):  # each sentence
            num_instance += 1
            #print(f"gt_moment:{gt_moment}") ## [start, end]
            #rint(f"pred_score2d:{pred_score2d.shape}") ## [64, 64]
            #print(f"pred_score2d:{pred_score2d}") ## [64, 64]
            candidates, scores = score2d_to_moments_scores(pred_score2d, num_clips, duration)
            moments = nms(candidates, scores, nms_thresh) ## only around 15% left

            for i, r in enumerate(recall_metrics): ## 1, 5
                mious = iou(moments[:r], gt_moment)
                bools = mious[:, None].expand(r, num_iou_metrics) >= iou_metrics
                recall_x_iou[i] += bools.any(dim=0)

    recall_x_iou /= num_instance
    table.append(['{:.02f}'.format(recall_x_iou[i][j]*100) for i in range(num_recall_metrics) for j in range(num_iou_metrics)])
    table = AsciiTable(table)

    for i in range(num_recall_metrics*num_iou_metrics):
        table.justify_columns[i] = 'center'
    logger.info('\n' + table.table)
    result_dict = {}
    for i in range(num_recall_metrics):
        for j in range(num_iou_metrics):
            result_dict['R@{},IoU@{:.01f}'.format(recall_metrics[i], torch.round(iou_metrics[j]*100)/100)] = recall_x_iou[i][j]
    best_r1 = sum(recall_x_iou[0])/num_iou_metrics
    best_r5 = sum(recall_x_iou[1])/num_iou_metrics
    result_dict['Best_R1'] = best_r1
    result_dict['Best_R5'] = best_r5
    return result_dict


def plot_prediction_and_GT(num_clips, dataset, idx, score2d, nms_result_locations=None, file_name=''):
    ## plot score2d
    combined_iou2d = dataset.get_iou2d(idx)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5.5))
    offset = torch.ones(num_clips, num_clips).triu()*0.05 ## for better visualization
    cm = plt.cm.get_cmap('Reds')

    ## plot prediction 2d map score
    iou_plot = axs[0].imshow(torch.squeeze(score2d)+offset, cmap=cm, vmin=0.0, vmax=1.0) ## score2d*3 for better visualization
    axs[0].set(xlabel='end index', ylabel='start index')
    axs[0].set_title(f"score2d")     
    divider = make_axes_locatable(axs[0])
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(iou_plot, cax=cax)

    ## plot nms results
    if nms_result_locations != None:
        for i, (row, col) in enumerate(nms_result_locations): 
            if i < 5:
                rect = plt.Rectangle((col-0.5, row-0.5), 1, 1, fill=False, color='green', linewidth=2)
                axs[0].add_patch(rect)
            elif i < 10: ## 5 <= i < 10
                rect = plt.Rectangle((col-0.5, row-0.5), 1, 1, fill=False, color='orange', linewidth=2)
                axs[0].add_patch(rect)    
            #else: ## i >= 10
            #    rect = plt.Rectangle((col-0.5, row-0.5), 1, 1, fill=False, color='black', linewidth=2, alpha=0.1)
            #    axs[0].add_patch(rect)

    ## plot combined_iou2d
    iou_plot = axs[1].imshow(torch.squeeze(combined_iou2d)+offset, cmap=cm, vmin=0.0, vmax=1.0)
    axs[1].set(xlabel='end index', ylabel='start index')
    axs[1].set_title(f"combined_iou2d")     
    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(iou_plot, cax=cax)

    ## plot nms results
    if nms_result_locations != None:
        for i, (row, col) in enumerate(nms_result_locations): 
            if i < 5:
                rect = plt.Rectangle((col-0.5, row-0.5), 1, 1, fill=False, color='green', linewidth=2)
                axs[1].add_patch(rect)
            elif i < 10: ## 5 <= i < 10
                rect = plt.Rectangle((col-0.5, row-0.5), 1, 1, fill=False, color='orange', linewidth=2)
                axs[1].add_patch(rect)    
            #else: ## i >= 10
            #    rect = plt.Rectangle((col-0.5, row-0.5), 1, 1, fill=False, color='black', linewidth=2, alpha=0.1)
            #    axs[1].add_patch(rect)

    

    ## settings
    fig.suptitle(f"{dataset.get_sentence(idx)}", y=0.9)
    fig.tight_layout()
    #plt.show()
    plt.savefig(file_name)
    plt.close()


## added for combined dataset
def evaluate_combined(cfg, dataset, predictions, nms_thresh, recall_metrics=(1, 5, 10)):
    ## predictions: {sample_idx: {'contrastive': result1, 'iou': result2}
    """evaluate dataset using different methods based on dataset type.
    Args:
    Returns:
    """
    if not is_main_process():
        return
    if cfg.DATASETS.NAME == "charades_combined":
        iou_metrics = (0.5, 0.7)
    else:
        raise NotImplementedError("No support for %s dataset!" % cfg.DATASETS.NAME)

    dataset_name = dataset.__class__.__name__
    logger = logging.getLogger("mmn.inference")
    logger.info("Performing {} evaluation (Size: {}).".format(dataset_name, len(dataset)))

    num_recall_metrics, num_iou_metrics = len(recall_metrics), len(iou_metrics)
    recall_metrics = torch.tensor(recall_metrics) ## (5)
    iou_metrics = torch.tensor(iou_metrics) ## (0.5, 0.7)
    num_clips = predictions[0]['matching_score'].shape[-1] ## 64
    table = [['R@{},IoU@{:.01f}'.format(i, torch.round(j*100)/100) for i in recall_metrics for j in iou_metrics]]
    table_1 = [['R@{},IoU@{:.01f}'.format(i, torch.round(j*100)/100) for i in recall_metrics for j in iou_metrics]]
    table_2 = [['R@{},IoU@{:.01f}'.format(i, torch.round(j*100)/100) for i in recall_metrics for j in iou_metrics]]
    table_3 = [['R@{},IoU@{:.01f}'.format(i, torch.round(j*100)/100) for i in recall_metrics for j in iou_metrics]]
    table_multi_target = [['R@{},IoU@{:.01f}'.format(i, torch.round(j*100)/100) for i in recall_metrics for j in iou_metrics]]

    recall_x_iou = torch.zeros(num_recall_metrics, num_iou_metrics) ## 
    recall_x_iou_1 = torch.zeros(num_recall_metrics, num_iou_metrics) ## 1 gt
    recall_x_iou_2 = torch.zeros(num_recall_metrics, num_iou_metrics) ## 2 gts
    recall_x_iou_3 = torch.zeros(num_recall_metrics, num_iou_metrics) ## 3 gts

    num_instance = 0 ## number of samples
    num_instance_1 = 0
    num_instance_2 = 0
    num_instance_3 = 0
    counter_1 = 0
    counter_2 = 0
    counter_3 = 0

    ## create folder for saving visualization results
    output_dir = cfg.OUTPUT_DIR
    if not os.path.exists(os.path.join(output_dir, 'plot')):
        os.makedirs(os.path.join(output_dir, 'plot'))

    result_record = {}
    ## predictions are sorted
    for idx, result2d in tqdm(enumerate(predictions)):   # each sample prediction
        ## only use predicted matching_score
        score2d = result2d['matching_score'] ## [num_query, 64, 64]
        
        duration = dataset.get_duration(idx) ## video length
        gt_moments = dataset.get_moment(idx) ## [[start1, end1], [start2, end2]...]
        num_gt = len(gt_moments)
        result_record[idx] = {"num_gt": num_gt}

        for _, pred_score2d in enumerate(score2d): ## iterate num_query in a sample (=1)
            num_instance += 1
            ## added
            if num_gt == 1:
                num_instance_1 += 1
            elif num_gt == 2:
                num_instance_2 += 1
            elif num_gt == 3:
                num_instance_3 +=1

            candidates, scores = score2d_to_moments_scores(pred_score2d, num_clips, duration)
            moments, remain_indices = nms(candidates, scores, nms_thresh) ## only around 15% left
           
            ## Plot Prediction Result
            nms_result_locations = pred_score2d.nonzero()[remain_indices]  ## for plotting      
            
            ## Save result plot
            #if idx % 1 == 0:
            #    file_name = os.path.join(output_dir, f'plot/{idx}.png')
            #    plot_prediction_and_GT(num_clips, dataset, idx, score2d, nms_result_locations, file_name)            
            

            for i, r in enumerate(recall_metrics): 

                '''
                ## Hungarian matching
                cost_matrix = np.ones((r, r))
                for j, gt_moment in enumerate(gt_moments):
                    mious = iou(moments[:r], gt_moment)
                    cost_matrix[j][:] = 1-mious 
                #print(f"iou matrix:\n{-cost_matrix+1}") ## row is GT, column is proposals

                ## row_ind: The row index of the cost matrix.  ex. [0, 1, 2, 3, 4]
                ## col_ind: The best assigned column index for the row index
                row_ind, col_ind = linear_sum_assignment(cost_matrix) ## hungarian algorithm
                row_ind, col_ind = row_ind[:num_gt], col_ind[:num_gt] ## corresponding to gt1, gt2, gt3...

                matching_iou = torch.tensor(-cost_matrix[row_ind, col_ind]+1) ## -(1-iou)+1 = iou
                #result_record[idx]['iou'] = matching_iou.tolist()  ## [0.88, 0.45]
                '''

                ## create a matrix: num_predictions x num_gts
                matching_iou = torch.zeros(r, len(gt_moments)) ##  top-k x num_gts 

                ## fill the matrix with iou values                
                for j, gt_moment in enumerate(gt_moments):
                    gt_ious = iou(moments[:r], gt_moment) ## the ious of top-k predictions and i-th gt moment
                    matching_iou[:, j] = gt_ious

                max_matching_iou = torch.max(matching_iou, dim=0)[0]

                ## record max matching iou
                result_record[idx]['iou'] = max_matching_iou.tolist()  ## [0.88, 0.45]


                for index, iou_metric in enumerate(iou_metrics):
                    metric_name = f"R@{r},IoU={round(iou_metric.item(), 1)}"
                    bools = matching_iou >= iou_metric
                    bools = torch.any(bools, dim=0) ## True if any proposal has iou > m with certain GT

                    recall_x_iou[i][index] += torch.count_nonzero(bools)/num_gt
                    result_record[idx][metric_name] = {'correct_count': torch.count_nonzero(bools).item(),
                                                       'result': bools.tolist()}

                    ## for each target number result
                    if num_gt == 1:
                        recall_x_iou_1[i][index] += torch.count_nonzero(bools)/num_gt
                        
                    elif num_gt == 2:
                        recall_x_iou_2[i][index] += torch.count_nonzero(bools)/num_gt
                        
                    elif num_gt == 3:
                        recall_x_iou_3[i][index] += torch.count_nonzero(bools)/num_gt

    recall_x_iou_multi_target = (recall_x_iou_2 + recall_x_iou_3) / (num_instance_2 + num_instance_3)
    recall_x_iou /= num_instance
    recall_x_iou_1 /= num_instance_1
    recall_x_iou_2 /= num_instance_2
    recall_x_iou_3 /= num_instance_3
    
    table.append(['{:.02f}'.format(recall_x_iou[i][j]*100) for i in range(num_recall_metrics) for j in range(num_iou_metrics)])
    table = AsciiTable(table)
    table_1.append(['{:.02f}'.format(recall_x_iou_1[i][j]*100) for i in range(num_recall_metrics) for j in range(num_iou_metrics)])
    table_1 = AsciiTable(table_1)
    table_2.append(['{:.02f}'.format(recall_x_iou_2[i][j]*100) for i in range(num_recall_metrics) for j in range(num_iou_metrics)])
    table_2 = AsciiTable(table_2)
    table_3.append(['{:.02f}'.format(recall_x_iou_3[i][j]*100) for i in range(num_recall_metrics) for j in range(num_iou_metrics)])
    table_3 = AsciiTable(table_3)
    table_multi_target.append(['{:.02f}'.format(recall_x_iou_multi_target[i][j]*100) for i in range(num_recall_metrics) for j in range(num_iou_metrics)])
    table_multi_target = AsciiTable(table_multi_target)

    for i in range(num_recall_metrics*num_iou_metrics):
        table.justify_columns[i] = 'center'
        table_1.justify_columns[i] = 'center'
        table_2.justify_columns[i] = 'center'
        table_3.justify_columns[i] = 'center'
        table_multi_target.justify_columns[i] = 'center'

    ## print the table
    logger.info('\n1 gt\n' + table_1.table)
    logger.info('\n2 gts\n' + table_2.table)
    logger.info('\n3 gts\n' + table_3.table)
    logger.info('\nAverage\n' + table.table)
    logger.info('\nMulti-target avg\n' + table_multi_target.table)
    #print(f"Num_instance:{num_instance}, 1:{num_instance_1}, 2:{num_instance_2}, 3:{num_instance_3}")

    result_dict = {}
    for i in range(num_recall_metrics):
        for j in range(num_iou_metrics):
            result_dict['R@{},IoU@{:.01f}'.format(recall_metrics[i], torch.round(iou_metrics[j]*100)/100)] = recall_x_iou[i][j]
    
    ## save result record
    with open(os.path.join(output_dir, 'result_record.json'), 'w') as f:
        json.dump(result_record, f)

    return result_dict


def get_top5_results(cfg, video_idx, dataset, predictions, nms_thresh, recall_metrics=(1, 5)):
    ## predictions: {video_idx: {'contrastive': result1, 'iou': result2}
    """evaluate dataset using different methods based on dataset type.
    Args:
    Returns:
    """
    if not is_main_process():
        return
    if cfg.DATASETS.NAME == "tacos":
        iou_metrics = (0.1, 0.3, 0.5)
    elif cfg.DATASETS.NAME == "activitynet":
        iou_metrics = (0.3, 0.5, 0.7)
    elif cfg.DATASETS.NAME == "charades":
        iou_metrics = (0.5, 0.7)
    else:
        raise NotImplementedError("No support for %s dataset!" % cfg.DATASETS.NAME)
    dataset_name = dataset.__class__.__name__
    logger = logging.getLogger("mmn.inference")
    logger.info("Performing {} evaluation (Size: {}).".format(dataset_name, len(dataset)))

    num_recall_metrics, num_iou_metrics = len(recall_metrics), len(iou_metrics)
    recall_metrics = torch.tensor(recall_metrics) ## (1, 5)
    iou_metrics = torch.tensor(iou_metrics) ## (0.3, 0.5, 0.7)
    #print(f"shape:{predictions[0]['iou'].shape}") ## [num_query, 64, 64]
    num_clips = predictions[0]['iou'].shape[-1] ## 64

    num_instance = 0
    for idx, result2d in tqdm(enumerate(predictions)):   # each video
        ## result2d['contrastive'] * 0.5 + 0.5: scale from [-1, 1] -> [0, 1]
        ## torch.pow(s_mm, 0.5): square root
        ## score_2d = s_mm^0.5 * s_iou
        result = []
        score2d = result2d['matching_score']
        #print(f"score_2d:{score2d.shape}") ## [num_query, 64, 64]
        duration = dataset.get_duration(video_idx) ## video length
        print(f"duration:{duration}")


        for i, pred_score2d in enumerate(score2d):  # each sentence
            num_instance += 1
            candidates, scores = score2d_to_moments_scores(pred_score2d, num_clips, duration)
            moments = nms(candidates, scores, nms_thresh)
            ## moments[:r] is top-r result
            result.append(moments[:5])

    return result

def get_new_video_top5_results(cfg, video_data_dict, predictions, nms_thresh):
    ## predictions: {video_idx: {'contrastive': result1, 'iou': result2}
    """evaluate dataset using different methods based on dataset type.
    Args:
    Returns:
    """
    if not is_main_process():
        return

    logger = logging.getLogger("mmn.inference")

    #print(f"shape:{predictions[0]['iou'].shape}") ## [num_query, 64, 64
    num_clips = predictions[0]['iou'].shape[-1] ## 64

    num_instance = 0
    for idx, result2d in tqdm(enumerate(predictions)):   # each video
        ## result2d['contrastive'] * 0.5 + 0.5: scale from [-1, 1] -> [0, 1]
        ## torch.pow(s_mm, 0.5): square root
        ## score_2d = s_mm^0.5 * s_iou
        result = []
        score2d = result2d['matching_score']
        #print(f"score_2d:{score2d.shape}") ## [num_query, 64, 64]
        duration = video_data_dict['duration'] ## video length


        for i, pred_score2d in enumerate(score2d):  # each sentence
            num_instance += 1
            candidates, scores = score2d_to_moments_scores(pred_score2d, num_clips, duration)
            moments = nms(candidates, scores, nms_thresh)
            ## moments[:r] is top-r result
            result.append(moments[:5])

    return result
