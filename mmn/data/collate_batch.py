import torch
from torch.nn.utils.rnn import pad_sequence
from mmn.structures import TLGBatch, TLGBatch_original



## sampler samples indices of a batch
## batch = self.collate_fn([self.dataset[i] for i in indices]) 
## ex. dataset[i] = (img, label), collate_fn needs to define how to organize the [(img, label), (img, label)....] 


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
        feats, sentences, queries, wordlens, ious2d, separate_iou2d, moments, num_sentence, \
        all_sentences, all_tokenized_queries, all_word_lens, idxs = transposed_batch
        
        return TLGBatch(
            feats=torch.stack(feats).float(),
            sentences=sentences,  ## original sentence
            queries=queries,      ## bert embeddings
            wordlens=wordlens,
            all_iou2d=ious2d,
            separate_iou2d=separate_iou2d,
            moments=moments,
            num_sentence=num_sentence,
            all_sentences=all_sentences,
            all_tokenized_queries=all_tokenized_queries,
            all_word_lens=all_word_lens,

        ), idxs

class BatchCollator_original(object):
    """
    Collect batch for dataloader
    """

    def __init__(self, ):
        pass

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        # [xxx, xxx, xxx], [xxx, xxx, xxx] ......
        feats, queries, wordlens, ious2d, moments, num_sentence, idxs = transposed_batch
        return TLGBatch_original(
            feats=torch.stack(feats).float(),
            queries=queries,
            wordlens=wordlens,
            all_iou2d=ious2d,
            moments=moments,
            num_sentence=num_sentence,
        ), idxs
