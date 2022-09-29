import logging
import torch
from mmn.data.datasets.evaluation import evaluate, evaluate_combined
from ..utils.comm import is_main_process, get_world_size
from ..utils.comm import all_gather
from ..utils.comm import synchronize
from ..utils.timer import Timer, get_time_str
from tqdm import tqdm

## added
from ..structures import TLGBatch
from mmn.data.datasets.evaluation import get_top5_results, get_new_video_top5_results



def compute_on_combined_dataset(model, data_loader, device, timer=None):
    model.eval()
    results_dict = {}
    cpu_device = torch.device("cpu")
    for batch in tqdm(data_loader):  # use tqdm(data_loader) for showing progress bar
        batches, idxs = batch
        #print(f"batches:{batches.feats.shape}") ## [batch_size, 256, 500]
        with torch.no_grad():
            if timer: ## True
                timer.tic()

            _, _, matching_output = model(batches.to(device))  ## ignore map2d_iou, sent_feat_iou
            
            if timer: ## True
                if not device.type == 'cpu':
                    torch.cuda.synchronize()
                timer.toc()
        
            matching_output = [o.to(cpu_device) for o in matching_output]
        
        results_dict.update(
            {sample_id: {'matching_score': score} for sample_id, score in zip(idxs, matching_output)}
        )

    return results_dict


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = all_gather(predictions_per_gpu) ## receiving Tensor from all ranks

    if not is_main_process(): ## not main process, then return
        return

    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    #print(f"keys:{predictions.keys()}")
    idxs = list(sorted(predictions.keys()))
    if len(idxs) != idxs[-1] + 1:
        logger = logging.getLogger("mmn.inference")
        logger.warning(
            "Number of samples that were gathered from multiple processes is not "
            "a contiguous set. Some samples might be missing from the evaluation"
        )

    # convert to a list
    predictions = [predictions[i] for i in idxs]
    return predictions


def inference(
        cfg,
        model,
        data_loader,
        dataset_name, 
        nms_thresh,
        device="cuda",
    ):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = get_world_size()
    logger = logging.getLogger("mmn.inference")
    dataset = data_loader.dataset
    logger.info("Start evaluation on {} dataset (Size: {}).".format(dataset_name, len(dataset)))
    inference_timer = Timer()

    ## model inference part
    ## predictions: result_dict: {video_id: {'contrastive': result1, 'iou': result2}
    predictions = compute_on_dataset(model, data_loader, device, inference_timer)
    # wait for all processes to complete before measuring the time
    synchronize()
    total_infer_time = get_time_str(inference_timer.total_time)
    
    logger.info(
        "Model inference time: {} ({:.03f} s / inference per device, on {} devices)".format(
            total_infer_time,
            inference_timer.total_time * num_devices / len(dataset),
            num_devices,
        )
    )

    predictions = _accumulate_predictions_from_multiple_gpus(predictions) ## for multiple gpus

    return evaluate(cfg, dataset=dataset, predictions=predictions, nms_thresh=nms_thresh)

def inference_specific_video(
        cfg,
        model,
        data_loader,
        video_name,
        dataset_name,
        nms_thresh,
        device="cuda",
    ):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = get_world_size()
    logger = logging.getLogger("mmn.inference")
    dataset = data_loader.dataset
    #print(f"dataset:{dataset}")
    logger.info("Start evaluation on {} dataset (Size: {}).".format(dataset_name, len(dataset)))
    inference_timer = Timer()

    ## get video_idx by name
    vid_to_idx_dict = dataset.video_name_to_index
    test_idx = vid_to_idx_dict[video_name]

    ## model inference part
    ## predictions: result_dict: {video_id: {'contrastive': result1, 'iou': result2}
    predictions = compute_specific_video(test_idx, model, dataset, device, inference_timer)
    # wait for all processes to complete before measuring the time
    synchronize()
    total_infer_time = get_time_str(inference_timer.total_time)
       
    logger.info(
        "Model inference time: {} ({:.03f} s / inference per device, on {} devices)".format(
            total_infer_time,
            inference_timer.total_time * num_devices / len(dataset),
            num_devices,
        )
    )

    predictions = _accumulate_predictions_from_multiple_gpus(predictions) ## for multiple gpus
    #_ = evaluate(cfg, dataset=dataset, predictions=predictions, nms_thresh=nms_thresh) ## nms = 0.5
    result = get_top5_results(cfg, video_idx=test_idx, dataset=dataset, predictions=predictions, nms_thresh=nms_thresh) ## nms = 0.5
    print(f"result:{result}")

    return result 

def inference_new_video(
        cfg,
        video_data_dict,
        model,
        nms_thresh,
        device="cuda",
    ):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = get_world_size()
    logger = logging.getLogger("mmn.inference")
    
    ''
    logger.info(f"Start evaluation on video {video_data_dict['video_name']}.")
    inference_timer = Timer()

    ## model inference part
    ## predictions: result_dict: {video_name: {'contrastive': result1, 'iou': result2}
    predictions = compute_new_video(video_data_dict, model, device, inference_timer)
    # wait for all processes to complete before measuring the time
    synchronize()

    total_infer_time = get_time_str(inference_timer.total_time)

    predictions = _accumulate_predictions_from_multiple_gpus(predictions) ## for multiple gpus

    result = get_new_video_top5_results(cfg, video_data_dict, predictions=predictions, nms_thresh=nms_thresh) ## nms = 0.5
        
    
    return result

def inference_combined_dataset(
        cfg,
        model,
        data_loader,
        dataset_name, 
        nms_thresh,
        device="cuda",
    ):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = get_world_size()
    logger = logging.getLogger("mmn.inference")
    dataset = data_loader.dataset
    logger.info("Start evaluation on {} dataset (Size: {}).".format(dataset_name, len(dataset)))
    inference_timer = Timer()

    ## model inference part
    ## predictions: result_dict: {video_id: {'contrastive': result1, 'iou': result2}
    predictions = compute_on_combined_dataset(model, data_loader, device, inference_timer)
    # wait for all processes to complete before measuring the time
    synchronize()
    total_infer_time = get_time_str(inference_timer.total_time)
    
    logger.info(
        "Model inference time: {} ({:.03f} s / inference per device, on {} devices)".format(
            total_infer_time,
            inference_timer.total_time * num_devices / len(dataset),
            num_devices,
        )
    )

    predictions = _accumulate_predictions_from_multiple_gpus(predictions) ## for multiple gpus

    return evaluate_combined(cfg, dataset=dataset, predictions=predictions, nms_thresh=nms_thresh) ## return the performance table