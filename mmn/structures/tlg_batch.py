from dataclasses import dataclass
import torch

# temporal localization grounding 
@dataclass
class TLGBatch(object):
    # frames: list # [ImageList]
    feats: torch.tensor 
    sentences: list  ## original sentence
    queries: list    ## bert embeddings
    wordlens: list
    all_iou2d: list
    separate_iou2d: list
    moments: list
    num_sentence: list

    ## added for intra_query loss
    all_sentences: list
    all_tokenized_queries: list
    all_word_lens: list

    def to(self, device):
        # self.frames = [f.to(device) for f in self.frames]
        self.sentences = [sentence[0] for sentence in self.sentences]
        self.feats = self.feats.to(device)
        self.queries = [query.to(device) for query in self.queries] ## only one query in combined dataset
        self.wordlens = [word_len.to(device) for word_len in self.wordlens]
        self.all_iou2d = [iou2d.to(device) for iou2d in self.all_iou2d] ## only one iou2d map in combined dataset
        self.separate_iou2d = [iou2d.to(device) for iou2ds in self.separate_iou2d for iou2d in iou2ds]
        self.moments = [moment.to(device) for moment in self.moments]

        ## added for intra_query_loss
        self.all_sentences = [sentence for all_sentence in self.all_sentences for sentence in all_sentence]
        self.all_tokenized_queries = [tokenized_query.unsqueeze(0).to(device) for all_tokenized_query in self.all_tokenized_queries for tokenized_query in all_tokenized_query]
        self.all_word_lens = [word_len.unsqueeze(0).to(device) for all_word_len in self.all_word_lens for word_len in all_word_len]

        return self

@dataclass
class TLGBatch_original(object):
    # frames: list # [ImageList]
    feats: torch.tensor 
    queries: list
    wordlens: list
    all_iou2d: list
    moments: list
    num_sentence: list

    def to(self, device):
        # self.frames = [f.to(device) for f in self.frames]
        self.feats = self.feats.to(device)
        self.queries = [query.to(device) for query in self.queries] ## only one query in combined dataset
        self.wordlens = [word_len.to(device) for word_len in self.wordlens]
        self.all_iou2d = [iou2d.to(device) for iou2d in self.all_iou2d] 
        self.moments = [moment.to(device) for moment in self.moments]

        return self


## TLGBatch for evaluating new raw video
@dataclass
class TLGBatch_new(object):
    # frames: list # [ImageList]
    feats: torch.tensor 
    queries: list
    wordlens: list
    num_sentence: list

    def to(self, device):
        self.feats = self.feats.to(device)
        self.queries = [query.to(device) for query in self.queries]
        self.wordlens = [word_len.to(device) for word_len in self.wordlens]

        return self