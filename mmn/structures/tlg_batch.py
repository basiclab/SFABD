from dataclasses import dataclass
import torch

# temporal localization grounding 
@dataclass
class TLGBatch(object):
    # frames: list # [ImageList]
    feats: torch.tensor 
    queries: list  ## original sentence
    tokenized_queries: list    ## bert tokens
    wordlens: list
    iou2d: list
    separate_iou2d: list
    moments: list
    num_target: list

    ## added for intra_query loss
    original_queries: list
    original_tokenized_queries: list
    original_word_lens: list

    def to(self, device):
        # self.frames = [f.to(device) for f in self.frames]
        self.queries = [query for query in self.queries]
        self.feats = self.feats.to(device)
        self.tokenized_queries = [query.to(device) for query in self.tokenized_queries] ## only one query in combined dataset
        self.wordlens = [word_len.to(device) for word_len in self.wordlens]
        self.iou2d = [iou2d.to(device) for iou2d in self.iou2d] ## only one iou2d map in combined dataset
        self.separate_iou2d = [iou2d.to(device) for iou2ds in self.separate_iou2d for iou2d in iou2ds]
        self.moments = [moment.to(device) for moment in self.moments]

        ## added for intra_query_loss
        ## each sample has multiple queries
        self.original_queries = [query for original_query in self.original_queries for query in original_query]
        self.original_tokenized_queries = [tokenized_query.unsqueeze(0).to(device) for original_tokenized_query in self.original_tokenized_queries for tokenized_query in original_tokenized_query]
        self.original_word_lens = [word_len.unsqueeze(0).to(device) for original_word_len in self.original_word_lens for word_len in original_word_len]

        return self
