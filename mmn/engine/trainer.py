import datetime
import logging
import os
import time
import gc
import torch
import torch.distributed as dist

from mmn.data import make_data_loader
from mmn.utils.comm import get_world_size, synchronize
from mmn.utils.metric_logger import MetricLogger
from mmn.engine.inference import inference, inference_combined_dataset
from ..utils.comm import is_main_process


def reduce_loss(loss):
    world_size = get_world_size()
    if world_size < 2:
        return loss
    with torch.no_grad():
        dist.reduce(loss, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            loss /= world_size
    loss = loss.item()
    return loss


def do_train(
    cfg,
    model,
    data_loader,
    data_loader_val,
    optimizer,
    scheduler,
    checkpointer,
    device,
    checkpoint_period,
    test_period,
    arguments,
    param_dict,
    max_norm=5
):

    logger = logging.getLogger("mmn.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_epoch = cfg.SOLVER.MAX_EPOCH

    model.train()
    start_training_time = time.time()
    end = time.time()
    max_iteration = len(data_loader)
    writer_count = 0

    for epoch in range(arguments["epoch"], max_epoch + 1):
        rest_epoch_iteration = (max_epoch - epoch) * max_iteration
        arguments["epoch"] = epoch
        #data_loader.batch_sampler.sampler.set_epoch(epoch) ## for distributed training
        
        ## Freeze BERT parameters
        if epoch <= cfg.SOLVER.FREEZE_BERT:
            for param in param_dict['bert']:
                param.requires_grad_(False)
        else:
            for param in param_dict['bert']:
                param.requires_grad_(True)


        logger.info("Start epoch {}. base_lr={:.1e}, bert_lr={:.1e}, bert.requires_grad={}".format(epoch, optimizer.param_groups[0]["lr"], optimizer.param_groups[1]["lr"], str(param_dict['bert'][0].requires_grad)))
        if epoch <= cfg.SOLVER.ONLY_IOU:
            logger.info("Using only bce loss")
        else:
            logger.info("Using all losses")

        ## batches is TLG batch
        for iteration, (batches, idx) in enumerate(data_loader):
            writer_count += 1
            iteration += 1
            batches = batches.to(device)
            optimizer.zero_grad()
            contr_weight = cfg.MODEL.MMN.LOSS.CONTRASTIVE_WEIGHT ## 0.05
            
            ## loss_vid, loss_sent: contrastive
            loss_vid, loss_sent, loss_intra_vid, loss_intra_query, loss_iou = model(batches, cur_epoch=epoch)
            loss_vid, loss_sent = loss_vid * contr_weight, loss_sent * contr_weight  ## contrastive_weight: 0.05
            loss_intra_vid = loss_intra_vid * contr_weight
            loss_intra_query = loss_intra_query * contr_weight
            
            #meters.update(loss_vid=loss_vid.detach(), loss_sent=loss_sent.detach(), loss_intra_vid=loss_intra_vid.detach(), 
            #              loss_iou=loss_iou.detach())
            meters.update(loss_vid=loss_vid.detach(), loss_sent=loss_sent.detach(), loss_intra_vid=loss_intra_vid.detach(), 
                          loss_intra_query=loss_intra_query.detach(), loss_iou=loss_iou.detach())

            #meters.update(loss_vid=loss_vid.detach(), loss_sent=loss_sent.detach(), loss_iou=loss_iou.detach())

            loss = 0
            if epoch <= cfg.SOLVER.ONLY_IOU: 
                loss += loss_iou
                #loss += loss_sent + loss_vid
            else: 
                loss += loss_iou
                #loss += (loss_sent + loss_vid) * 0.01
                #loss += (loss_sent + loss_vid) * 0.1 ## baseline
                #loss += (loss_sent + loss_vid + loss_intra_vid) * 0.1
                #loss += (loss_sent + loss_vid + loss_intra_query) * 0.1
                loss += (loss_sent + loss_vid + loss_intra_vid + loss_intra_query) * 0.1

            loss.backward()

            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()

            batch_time = time.time() - end
            end = time.time()
            meters.update(time=batch_time)
            eta_seconds = meters.time.global_avg * (max_iteration - iteration + rest_epoch_iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            if iteration % 10 == 0 or iteration == max_iteration:
                logger.info(
                    meters.delimiter.join(
                        [
                            "eta: {eta}",
                            "epoch: {epoch}/{max_epoch}",
                            "iteration: {iteration}/{max_iteration}",
                            "{meters}",
                            "max mem: {memory:.0f}",
                        ]
                    ).format(
                        eta=eta_string,
                        epoch=epoch,
                        max_epoch=max_epoch,
                        iteration=iteration,
                        max_iteration=max_iteration,
                        meters=str(meters),
                        memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                    )
                )
            gc.collect()

        scheduler.step()
        if checkpoint_period != -1 and epoch % checkpoint_period == 0:
            checkpointer.save(f"{cfg.MODEL.MMN.FEAT2D.NAME}_model_{epoch}e", **arguments) ## pool_model_2e.pth

        ## Evaluate training set
        synchronize()
        torch.cuda.empty_cache()

        result_dict = inference_combined_dataset(
            cfg,
            model,
            data_loader,
            dataset_name=cfg.DATASETS.TRAIN,
            nms_thresh=cfg.TEST.NMS_THRESH,
            device=cfg.MODEL.DEVICE,
        )
        synchronize()
        model.train()


        ## Evaluation
        if data_loader_val is not None and test_period > 0 and epoch % test_period == 0 and epoch >= cfg.SOLVER.SKIP_TEST: ## test_period=1
            synchronize()
            torch.cuda.empty_cache()

            result_dict = inference_combined_dataset(
                cfg,
                model,
                data_loader_val,
                dataset_name=cfg.DATASETS.TEST,
                nms_thresh=cfg.TEST.NMS_THRESH,
                device=cfg.MODEL.DEVICE,
            )
            synchronize()
            model.train()
            

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iteration)
        )
    )
