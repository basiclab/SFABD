import torch
from torch.nn.utils.rnn import pad_sequence
from mmn.structures import TLGBatch


## custom collate_batch
class BatchCollator(object):
    """
    Collect batch for dataloader
    """

    def __init__(self, ):
        pass

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        # [xxx, xxx, xxx], [xxx, xxx, xxx] ......
        feats, \
        queries, \
        tokenized_queries, \
        wordlens, \
        ious2d, \
        separate_iou2d, \
        moments, \
        num_target, \
        original_queries, \
        original_tokenized_queries, \
        original_word_lens, \
        idxs = transposed_batch \
        
        return TLGBatch(
            feats=torch.stack(feats).float(),
            queries=queries,  ## original sentence
            tokenized_queries=tokenized_queries,      ## bert embeddings
            wordlens=wordlens,
            iou2d=ious2d,
            separate_iou2d=separate_iou2d,
            moments=moments,
            num_target=num_target,
            original_queries=original_queries,
            original_tokenized_queries=original_tokenized_queries,
            original_word_lens=original_word_lens,

        ), idxs
