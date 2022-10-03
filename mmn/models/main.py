import torch
import torch.nn as nn
import torch.nn.functional as F

from mmn.models.text import LanguageModel
from mmn.models.proposal import ProposalConv


class Conv1dPool(nn.Module):
    def __init__(self, in_channel, out_channel, pool_kernel_size, pool_stride_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, 1, 1),
            nn.ReLU(inplace=False),
            nn.AvgPool1d(pool_kernel_size, pool_stride_size),
        )

    def forward(
        self,
        x: torch.Tensor         # [B, C, NUM_INIT_CLIPS]
    ):
        """
            B: (B)atch size
            C: (C)hannel = JOINT_SPACE_SIZE
            D: (D)imension of prorposal matrix = NUM_CLIPS
        """
        x = self.model(x)       # [B, C, D]
        return x


class ProposalPool(nn.Module):
    def forward(self, x: torch.Tensor):
        """
            B: (B)atch size
            C: (C)hannel = JOINT_SPACE_SIZE
            D: (D)imension of prorposal matrix = NUM_CLIPS
        """
        B, C, D = x.shape
        zero = x.new_zeros(B, C)
        x2d = []
        for i in range(D):
            for j in range(D):
                if i <= j:
                    x2d.append(x[:, :, i: j + 1].max(dim=2).values)
                else:
                    x2d.append(zero)

        x2d = torch.stack(x2d, dim=-1).view(B, C, D, D)
        return x2d


class MMN(nn.Module):
    def __init__(
        self,
        conv1d_in_channel: int,         # Conv1dPool
        conv1d_out_channel: int,        # Conv1dPool
        conv1d_pool_kerenl_size: int,   # Conv1dPool
        conv1d_pool_stride_size: int,   # Conv1dPool
        conv2d_in_dim: int,             # ProposalConv
        conv2d_in_channel: int,         # ProposalConv
        conv2d_hidden_channel: int,     # ProposalConv
        conv2d_kernel_size: int,        # ProposalConv
        conv2d_num_layers: int,         # ProposalConv
        joint_space_size: int,
    ):
        """
            B: (B)atch size
            C: (C)hannel = JOINT_SPACE_SIZE
            D: (D)imension of prorposal matrix = NUM_CLIPS
        """
        super(MMN, self).__init__()
        self.video_model = nn.Sequential(
            Conv1dPool(                                     # [B, C, NUM_INIT_CLIPS]
                conv1d_in_channel,
                conv1d_out_channel,
                conv1d_pool_kerenl_size,
                conv1d_pool_stride_size,
            ),                                              # [B, C, D]
            ProposalPool(),                                 # [B, C, D, D]
            ProposalConv(
                conv2d_in_dim,
                conv2d_in_channel,
                conv2d_hidden_channel,
                joint_space_size,
                conv2d_kernel_size,
                conv2d_num_layers,
            )                                               # [B, C, D, D]
        )
        self.query_model = LanguageModel(joint_space_size)  # [B, C]

    def matching_scores(self, video_feats, query_feats):
        """
        Return matching scores between each proposal and corresponding query.

        Return:
            scores_matrix: [B, D, D], the value range is [-1, 1]
        """
        B, C, D, _ = video_feats.shape
        device = video_feats.device
        ones = torch.ones(D, D, device=device).bool()  # [D, D]
        mask = torch.triu(ones, diagonal=0)            # [D, D]
        video_feats = video_feats.masked_select(mask).view(B, C, -1)    # [B, C, P]
        video_feats = video_feats.permute(0, 2, 1)                      # [B, P, C]
        video_feats = F.normalize(video_feats, dim=-1)                  # [B, P, C]
        query_feats = F.normalize(query_feats, dim=-1)                  # [B, C]
        scores = torch.mul(video_feats, query_feats.unsqueeze(1))       # [B, P, C]
        scores = scores.sum(dim=-1)                                     # [B, P]

        scores_matrix = torch.zeros(B, D, D, device=device)             # [B, D, D]
        scores_matrix[mask.unsqueeze(0).expand(B, -1, -1)] = scores.view(-1)
        scores_matrix = torch.sigmoid(scores_matrix * 10)               # TODO: magic
        return scores_matrix                                            # [B, D, D]

    def forward(
        self,
        video_feats: torch.Tensor,          # [B, C, NUM_INIT_CLIPS]
        query_tokens: torch.Tensor,         # [B, L]
        query_length: torch.Tensor,         # [B]
        sents_tokens: torch.Tensor,         # [T, L]
        sents_length: torch.Tensor,         # [T]
        **kwargs,                           # dummy
    ):
        """
            B: (B)atch size
            C: (C)hannel = JOINT_SPACE_SIZE
            D: (D)imension of prorposal matrix = NUM_CLIPS
            T: number of (T)argets in batch
            L: (L)ength of tokens
        """
        video_feats = self.video_model(video_feats)
        query_feats = self.query_model(query_tokens, query_length)
        sents_feats = self.query_model(sents_tokens, sents_length)
        scores = self.matching_scores(video_feats, query_feats)

        return video_feats, query_feats, sents_feats, scores

    def evaluate(
        self,
        video_feats: torch.Tensor,          # [B, NUM_INIT_CLIPS, C]
        query_tokens: torch.Tensor,         # [B, L]
        query_length: torch.Tensor,         # [B]
        **kwargs,                           # dummy
    ):
        """
            B: (B)atch size
            C: (C)hannel = JOINT_SPACE_SIZE
            L: (L)ength of tokens
        """
        video_feats = self.video_model(video_feats)
        query_feats = self.query_model(query_tokens, query_length)
        scores = self.matching_scores(video_feats, query_feats)

        return video_feats, query_feats, None, scores


if __name__ == '__main__':
    from transformers import DistilBertTokenizer

    B = 8
    INIT_CHANNEL = 4096
    NUM_INIT_CLIPS = 64
    NUM_CLIPS = 32
    JOINT_DIM = 256

    model = MMN(
        conv1d_in_channel=INIT_CHANNEL,
        conv1d_out_channel=512,
        conv1d_pool_kerenl_size=2,
        conv1d_pool_stride_size=NUM_INIT_CLIPS // NUM_CLIPS,
        conv2d_in_dim=NUM_CLIPS,
        conv2d_in_channel=512,
        conv2d_hidden_channel=512,
        conv2d_kernel_size=5,
        conv2d_num_layers=8,
        joint_space_size=JOINT_DIM
    )

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    query = [
        "a person is putting a book on a shelf.",
        "person begins to play on a phone.",
        "a person a person closes the door",
        "a person a person closes the door",
        "a person a person is looking outside of a window",
        "person eats the sandwich.",
        "a person a person is eating something",
        "a person a person is sitting somewhere",
    ]
    sents = [
        "a person is putting a book on a shelf.",
        "person begins to play on a phone.",
        "person closes the door behind them.",
        "the person closes the dryer door.",
        "person closes the frig door.",
        "a person closes the door.",
        "a person is looking out a window.",
        "person standing to look out the window.",
        "another person is looking out a window.",
        "person eats the sandwich.",
        "a person is sitting at at table eating a sandwich.",
        "person eating something.",
        "person eating something.",
        "person sitting on the floor.",
        "a person sitting in a chair takes off their shoes.",
    ]
    num_targets = [1, 1, 2, 2, 3, 1, 3, 2]
    T = sum(num_targets)
    assert T == len(sents)
    query = tokenizer(
        query, padding=True, return_tensors="pt", return_length=True)
    sents = tokenizer(
        sents, padding=True, return_tensors="pt", return_length=True)

    video_feats = torch.randn(B, 4096, NUM_INIT_CLIPS)
    query_tokens = query['input_ids']
    query_lengthgth = query['length']
    sents_tokens = sents['input_ids']
    sents_lengthgth = sents['length']

    video_feats, query_feats, sents_feats, scores = model(
        video_feats, query_tokens, query_lengthgth, sents_tokens, sents_lengthgth)

    print(f"video_feats   : {video_feats.shape}")
    print(f"query_feats   : {query_feats.shape}")
    print(f"sentence_feats: {sents_feats.shape}")
    print(f"scores        : {scores.shape}")

    assert video_feats.shape == (B, JOINT_DIM, NUM_CLIPS, NUM_CLIPS)
    assert query_feats.shape == (B, JOINT_DIM)
    assert sents_feats.shape == (T, JOINT_DIM)
    assert scores.shape == (B, NUM_CLIPS, NUM_CLIPS)
