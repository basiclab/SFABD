import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertModel
from transformers import logging


class AggregateVideo(nn.Module):
    def __init__(self, tgt_num: int):
        super().__init__()
        self.tgt_num = tgt_num

    def aggregate_feats(
        self,
        video_feats: torch.Tensor,  # [src_num, C]
    ) -> torch.Tensor:              # [tgt_num, C]
        """Aggregate the feature of video into fixed shape."""

        src_num, _ = video_feats.shape
        idxs = torch.arange(0, self.tgt_num + 1) / self.tgt_num * src_num
        idxs = idxs.round().long().clamp(max=src_num - 1)
        feats_bucket = []
        for i in range(self.tgt_num):
            s, e = idxs[i], idxs[i + 1]
            if s < e:
                feats_bucket.append(video_feats[s:e].mean(dim=0))
            else:
                feats_bucket.append(video_feats[s])
        return torch.stack(feats_bucket)

    def forward(
        self,
        video_feats: torch.Tensor,          # [B, T, C]
        video_masks: torch.Tensor,          # [B, T]
    ) -> torch.Tensor:                      # [B, tgt_num, C]
        out_feats = []
        for i in range(len(video_feats)):
            out_feat = self.aggregate_feats(video_feats[i][video_masks[i]])
            out_feats.append(out_feat)
        out_feats = torch.stack(out_feats)
        return out_feats


class Conv1dPool(nn.Module):
    def __init__(self, in_channel, out_channel, pool_kernel_size, pool_stride_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, 1, 1),
            nn.ReLU(inplace=True),
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


class SparseMaxPool(nn.Module):
    def __init__(self, counts):
        super().__init__()
        self.counts = counts

    def forward(self, x):
        B, C, N = x.shape
        mask2d = torch.eye(N, N, device=x.device).bool()
        x2d = x.new_zeros(B, C, N, N)

        stride, offset = 1, 0
        for level, count in enumerate(self.counts):
            if level != 0:
                x = torch.nn.functional.max_pool1d(x, 3, 2)
            for order in range(count):
                if order != 0:
                    x = torch.nn.functional.max_pool1d(x, 2, 1)
                i = range(0, N - offset, stride)
                j = range(offset, N, stride)
                x2d[:, :, i, j] = x
                mask2d[i, j] = 1
                offset += stride
            offset += stride
            stride *= 2
        return x2d, mask2d


class SparsePropConv(nn.Module):
    def __init__(self, counts, hidden_size):
        super().__init__()
        self.num_scale_layers = counts
        self.hidden_size = hidden_size
        self.convs = nn.ModuleList()
        self.convs.extend(
                [nn.Conv1d(hidden_size, hidden_size, 1, 1)] + [nn.Conv1d(hidden_size, hidden_size, 2, 1) for _ in range(self.num_scale_layers[0]-1)]
            )  
        for count in self.num_scale_layers[1:]: ## start from 2nd layer
            self.convs.extend(
                #[nn.Conv1d(hidden_size, hidden_size, 3, 2)] + [nn.Conv1d(hidden_size, hidden_size, 2, 1) for _ in range(count-1)]
                [nn.MaxPool1d(3, 2)] + [nn.Conv1d(hidden_size, hidden_size, 2, 1) for _ in range(count-1)]
            )        
    
    def forward(self, x):
        B, C, N = x.shape
        mask2d = torch.eye(N, N, device=x.device).bool()
        x2d = x.new_zeros(B, C, N, N)

        ## offset: for which diagonal line on 2D map
        ## stride: interval of proposals to put on 2D map    
        accumulate_count = 0
        stride, offset = 1, 0
        for level, count in enumerate(self.num_scale_layers): ## (0, 16),  (1, 8), (1, 8)
            for order in range(count):
                x = self.convs[accumulate_count](x)
                x = F.relu(x)

                i = range(0, N - offset, stride)
                j = range(offset, N, stride)
                x2d[:, :, i, j] = x
                mask2d[i, j] = 1
                offset += stride ## offset for diagonal line
                accumulate_count += 1

            offset += stride
            stride *= 2

        return x2d, mask2d



class SparseConvShareWeight(nn.Module):
    def __init__(self, counts, hidden_size, num_clips):
        super().__init__()
        self.counts = counts ## list
        self.conv1 = nn.Conv1d(hidden_size, hidden_size, 1, 1)
        self.convs = nn.ModuleList()
        self.convs.extend([nn.Conv1d(hidden_size, hidden_size, 2, 1), 
                           nn.Conv1d(hidden_size, hidden_size, 2, 1),
                           nn.Conv1d(hidden_size, hidden_size, 2, 1)])
        #self.boundary_convs = nn.ModuleList()
        #self.boundary_convs.extend([nn.Conv1d(hidden_size, hidden_size, 3, 2),
        #                            nn.Conv1d(hidden_size, hidden_size, 3, 2)])

        #self.duration_encoding = nn.Embedding(num_clips, hidden_size) ## for encoding proposal length 


    def forward(self, x):
        B, C, N = x.shape
        mask2d = torch.eye(N, N, device=x.device).bool()
        x2d = x.new_zeros(B, C, N, N)

        ## offset: for which diagonal line on 2D map
        ## stride: interval of proposals to put on 2D map    
        stride, offset = 1, 0  
        for level, count in enumerate(self.counts): ## (0, 16),  (1, 8), (1, 8)
            for order in range(count):
                i = range(0, N - offset, stride)
                j = range(offset, N, stride)

                proposal_duration = torch.tensor([j[0] - i[0]]) ## the duration before conv
                proposal_duration = proposal_duration.to(x.device)

                ## first
                if (level == 0 and order == 0):
                  x = self.conv1(x)
                ## not first
                else:
                    if order == 0:
                        x = torch.nn.functional.max_pool1d(x, 3, 2)
                        #x = self.boundary_convs[level-1](x)
                    else:
                        x = self.convs[level](x)
                        #x = self.convs[level](x + self.duration_encoding(proposal_duration).t())

                x2d[:, :, i, j] = x 
                mask2d[i, j] = 1
                offset += stride ## offset for diagonal line

            offset += stride
            stride *= 2

        return x2d, mask2d


class ProposalConv(nn.Module):
    def __init__(
        self,
        in_channel: int,            # input feature size
        hidden_channel: int,        # hidden feature size
        out_channel: int,           # output feature size
        kernel_size: int,           # kernel size
        num_layers: int,            # number of CNN layers (exclude the projection layers)
        dual_scpae: bool = False    # whether to use dual feature scpace
    ):
        super(ProposalConv, self).__init__()
        self.kernel_size = kernel_size
        self.dual_scpae = dual_scpae

        self.blocks = nn.ModuleList()
        self.paddings = []
        for idx in range(num_layers):
            if idx == 0:
                padding = (kernel_size - 1) * num_layers // 2
                channel = in_channel
            else:
                padding = 0
                channel = hidden_channel
            self.blocks.append(nn.Sequential(
                nn.Conv2d(
                    channel, hidden_channel, kernel_size, padding=padding),
                nn.BatchNorm2d(hidden_channel),
                nn.ReLU(inplace=True),
            ))
            self.paddings.append(padding)

        if dual_scpae:
            self.proj1 = nn.Conv2d(hidden_channel, out_channel, 1)
            self.proj2 = nn.Conv2d(hidden_channel, out_channel, 1)
        else:
            self.proj = nn.Conv2d(hidden_channel, out_channel, 1)

    def get_masked_weight(self, mask, padding):
        masked_weight = torch.round(F.conv2d(
            mask.float(),
            mask.new_ones(1, 1, self.kernel_size, self.kernel_size).float(),
            padding=padding))
        masked_weight[masked_weight > 0] = 1 / masked_weight[masked_weight > 0]
        mask = masked_weight > 0
        return mask, masked_weight

    def forward(self, input):
        x, mask2d = input
        mask = mask2d.detach().clone().unsqueeze(0).unsqueeze(0)
        for padding, block in zip(self.paddings, self.blocks):
            mask, masked_weight = self.get_masked_weight(mask, padding)
            x = block(x) * masked_weight
        if self.dual_scpae:
            x1 = self.proj1(x)
            x2 = self.proj2(x)
        else:
            x1 = self.proj(x)
            x2 = x1
        return x1, x2, mask2d

class ProposalConv_PE(nn.Module):
    def __init__(
        self,
        in_channel: int,            # input feature size
        hidden_channel: int,        # hidden feature size
        out_channel: int,           # output feature size
        kernel_size: int,           # kernel size
        num_layers: int,            # number of CNN layers (exclude the projection layers)
    ):
        super(ProposalConv, self).__init__()
        self.kernel_size = kernel_size

        self.blocks = nn.ModuleList()
        self.paddings = []
        for idx in range(num_layers):
            if idx == 0:
                padding = (kernel_size - 1) * num_layers // 2
                channel = in_channel
            else:
                padding = 0
                channel = hidden_channel
            self.blocks.append(nn.Sequential(
                nn.Conv2d(
                    channel, hidden_channel, kernel_size, padding=padding),
                nn.BatchNorm2d(hidden_channel),
                nn.ReLU(inplace=True),
            ))
            self.paddings.append(padding)

            self.mean_proj = nn.Conv2d(hidden_channel, out_channel, 1)
            self.log_sigma_proj = nn.Conv2d(hidden_channel, out_channel, 1) 


    def get_masked_weight(self, mask, padding):
        masked_weight = torch.round(F.conv2d(
            mask.float(),
            mask.new_ones(1, 1, self.kernel_size, self.kernel_size).float(),
            padding=padding))
        masked_weight[masked_weight > 0] = 1 / masked_weight[masked_weight > 0]
        mask = masked_weight > 0
        return mask, masked_weight

    def forward(self, input):
        x, mask2d = input
        mask = mask2d.detach().clone().unsqueeze(0).unsqueeze(0)
        for padding, block in zip(self.paddings, self.blocks):
            mask, masked_weight = self.get_masked_weight(mask, padding)
            x = block(x) * masked_weight

            x_mean = self.mean_proj(x)
            x_log_sigma = self.log_sigma_proj(x)

        return x_mean, x_log_sigma, mask2d



class LanguageModel(nn.Module):
    def __init__(self, joint_space_size, dual_scpae=False):
        super().__init__()
        self.dual_scpae = dual_scpae

        logging.set_verbosity_error()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        logging.set_verbosity_warning()

        if dual_scpae:
            self.proj1 = nn.Sequential(
                nn.LayerNorm(768),
                nn.Linear(768, joint_space_size),
            )
            self.proj2 = nn.Sequential(
                nn.LayerNorm(768),
                nn.Linear(768, joint_space_size),
            )
        else:
            self.proj = nn.Sequential(
                nn.LayerNorm(768),
                nn.Linear(768, joint_space_size),
            )

    def forward(
        self,
        sents_tokens: torch.Tensor,                                         # [S, L]
        sents_masks: torch.Tensor,                                          # [S, L]
    ):
        feats = self.bert(sents_tokens, attention_mask=sents_masks)[0]      # [S, L, C]
        feats = (feats * sents_masks.unsqueeze(-1)).sum(dim=1)              # [S, C]
        feats = feats / sents_masks.sum(dim=1, keepdim=True)                # [S, C]

        if self.dual_scpae:
            feats1 = self.proj1(feats)                                      # [S, C]
            feats2 = self.proj2(feats)                                      # [S, C]
        else:
            feats1 = self.proj(feats)                                       # [S, C]
            feats2 = feats1                                                 # [S, C]
        return feats1, feats2


class LanguageModel_PE(nn.Module):
    def __init__(self, joint_space_size):
        super().__init__()

        logging.set_verbosity_error()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        logging.set_verbosity_warning()

        self.mean_proj = nn.Sequential(
                nn.LayerNorm(768),
                nn.Linear(768, joint_space_size),
        )

        self.log_sigma_proj = nn.Sequential(
                nn.LayerNorm(768),
                nn.Linear(768, joint_space_size),
        )

    def forward(
        self,
        sents_tokens: torch.Tensor,                                         # [S, L]
        sents_masks: torch.Tensor,                                          # [S, L]
    ):
        feats = self.bert(sents_tokens, attention_mask=sents_masks)[0]      # [S, L, C]
        feats = (feats * sents_masks.unsqueeze(-1)).sum(dim=1)              # [S, C]
        feats = feats / sents_masks.sum(dim=1, keepdim=True)                # [S, C]

        mean_feats = self.mean_proj(feats)                                  # [S, C]
        log_sigma_feats = self.log_sigma_proj(feats)                        # [S, C]
        
        return mean_feats, log_sigma_feats
