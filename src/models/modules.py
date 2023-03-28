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
            #print(f"s:{s}, e:{e}")
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
        #self.proj = nn.Conv2d(512, 256, 1)

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
        #return x2d, x2d, mask2d

class SparsePropConv(nn.Module):
    def __init__(self, counts, hidden_size):
        super().__init__()
        self.num_scale_layers = counts
        self.hidden_size = hidden_size
        # torch.nn.init.normal_(self.proj.weight, std=0.2)
        self.convs = nn.ModuleList()
        
        for layer_idx, layer_count in enumerate(self.num_scale_layers):
            ## first layer
            ## no first conv1d
            if layer_idx == 0:
                for order in range(layer_count):
                    if order != 0:
                        if (order % 2) != 0: ## 1, 3, 5 ...
                            # self.convs.extend([nn.MaxPool1d(2, 1)])
                            self.convs.extend([
                                nn.Sequential(
                                    nn.Conv1d(hidden_size, hidden_size, 2, 1),
                                    nn.BatchNorm1d(hidden_size),
                                    nn.ReLU(),
                                )
                            ])
                            
                        else: ## 2, 4, 6 ...
                            # self.convs.extend([
                            #     nn.Sequential(
                            #         nn.Conv1d(hidden_size, hidden_size, 2, 1),
                            #         nn.BatchNorm1d(hidden_size),
                            #         nn.ReLU(),
                            #     )
                            # ])
                            self.convs.extend([nn.MaxPool1d(2, 1)])
            ## with first conv1d
            # if layer_idx == 0:
            #     for order in range(layer_count):
            #         if order == 0:
            #             self.convs.extend([nn.Conv1d(hidden_size, hidden_size, 1, 1)])
            #         else:
            #             if (order % 2) != 0: ## 1, 3, 5 ...
            #                 self.convs.extend([nn.MaxPool1d(2, 1)])
                            
            #             else: ## 2, 4, 6 ...
            #                 self.convs.extend([
            #                     nn.Sequential(
            #                         nn.Conv1d(hidden_size, hidden_size, 2, 1),
            #                         nn.BatchNorm1d(hidden_size),
            #                         nn.ReLU(),
            #                     )
            #                 ])

            ## other layers 
            else: 
                for order in range(layer_count):
                    if order == 0:
                        #self.convs.extend([nn.Conv1d(hidden_size, hidden_size, 3, 2)])
                        self.convs.extend([nn.MaxPool1d(3, 2)])
                    
                    else:    
                        if (order % 2) != 0: ## 1, 3, 5 ...
                            # self.convs.extend([nn.MaxPool1d(2, 1)])
                            self.convs.extend([
                                nn.Sequential(
                                    nn.Conv1d(hidden_size, hidden_size, 2, 1),
                                    nn.BatchNorm1d(hidden_size),
                                    nn.ReLU(),
                                )
                            ])
                        else: ## 2, 4, 6 ...
                            # self.convs.extend([
                            #     nn.Sequential(
                            #         nn.Conv1d(hidden_size, hidden_size, 2, 1),
                            #         nn.BatchNorm1d(hidden_size),
                            #         nn.ReLU(),
                            #     )
                            # ])
                            self.convs.extend([nn.MaxPool1d(2, 1)])
                        
                # self.convs.extend([nn.MaxPool1d(3, 2)])
                # self.convs.extend([nn.Sequential(
                #     nn.Conv1d(hidden_size, hidden_size, 2, 1),
                #     nn.BatchNorm1d(hidden_size),
                #     nn.ReLU(),
                # ) for _ in range(layer_count-1)])            

    def forward(self, x):
        B, C, N = x.shape
        mask2d = torch.eye(N, N, device=x.device).bool()
        x2d = x.new_zeros(B, C, N, N)

        ## offset: for which diagonal line on 2D map
        ## stride: interval of proposals to put on 2D map    
        accumulate_count = 0
        stride, offset = 1, 0
        for level, count in enumerate(self.num_scale_layers): ## (0, 16), (1, 8), (2, 8)
            for order in range(count):
                ## for no initial layer
                if accumulate_count != 0:
                    x = self.convs[accumulate_count-1](x)
                #x = self.convs[accumulate_count](x)               

                i = range(0, N - offset, stride)
                j = range(offset, N, stride)
                x2d[:, :, i, j] = x
                mask2d[i, j] = 1
                offset += stride ## offset for diagonal line
                accumulate_count += 1

            offset += stride
            stride *= 2

        return x2d, mask2d

class ProposalConv(nn.Module):
    def __init__(
        self,
        in_channel: int,            # input feature size (512)
        hidden_channel: int,        # hidden feature size (512)
        out_channel: int,           # output feature size (256)
        kernel_size: int,           # kernel size
        num_layers: int,            # number of CNN layers (exclude the projection layers)
        dual_space: bool = False    # whether to use dual feature scpace
    ):
        super(ProposalConv, self).__init__()
        self.kernel_size = kernel_size
        self.dual_space = dual_space

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

        if dual_space:
            self.proj1 = nn.Conv2d(hidden_channel, out_channel, 1) ## 512 -> 256
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
            
        if self.dual_space:
            x1 = self.proj1(x)
            x2 = self.proj2(x)
        else:
            x1 = self.proj(x)
            x2 = x1
            
        return x1, x2, mask2d

class LanguageModel(nn.Module):
    def __init__(self, joint_space_size, dual_space=False):
        super().__init__()
        self.dual_space = dual_space

        logging.set_verbosity_error()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        logging.set_verbosity_warning()

        if dual_space:
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

        if self.dual_space:
            feats1 = self.proj1(feats)                                      # [S, C]
            feats2 = self.proj2(feats)                                      # [S, C]
        else:
            feats1 = self.proj(feats)                                       # [S, C]
            feats2 = feats1                                                 # [S, C]
        return feats1, feats2

## Bbox regression
class BboxRegression(nn.Module):
    def __init__(
        self,
        in_channel: int,        # dim of embedding space
    ):
        super().__init__()
        # self.bbox_predictor = nn.Sequential(
        #                             nn.Conv2d(in_channel*2, in_channel, 1),
        #                             nn.ReLU(),
        #                             #nn.Dropout(0.5),
        #                             nn.Conv2d(in_channel, 2, 1),
        #                         )

        self.start_offset_head = nn.Sequential(
                                    nn.Conv2d(in_channel*3, in_channel, 1),
                                    nn.ReLU(),
                                    #nn.Dropout(0.5),
                                    nn.Conv2d(in_channel, 1, 1),
                                )

        self.end_offset_head = nn.Sequential(
                                    nn.Conv2d(in_channel*3, in_channel, 1),
                                    nn.ReLU(),
                                    #nn.Dropout(0.5),
                                    nn.Conv2d(in_channel, 1, 1),
                                )


    def forward(
        self, 
        video_feats,  ## [S, C, N, N] 
        sent_feats,   ## [S, C]
    ):
        N = video_feats.shape[-1]
        S, C = sent_feats.shape
        sent_feats = sent_feats.view(S, C, 1, 1)        ## [S, C, 1, 1]
        sent_feats = sent_feats.expand(-1, -1, N, N)    ## [S, C, N, N]
        
        ## TODO: Also consider surrounding clips?
        # index the diagonal proposal features (len=1)
        unit_clips = torch.diagonal(video_feats, offset=0, dim1=-2, dim2=-1)    ## [S, C, N]
        # prepare start extra clips for start offset head
        start_clips = unit_clips.unsqueeze(-1).expand(-1, -1, -1, N)            ## [S, C, N, N]
        zero_pad = torch.zeros([S, C, 1, N], device=video_feats.device)
        start_clips = torch.cat((zero_pad, start_clips), dim=-2)                ## [S, C, 1 + N, N]
        start_clips = start_clips[:, :, :N, :]                                  ## [S, C, N, N]
        
        # prepare end extra clips for start offset head
        end_clips = unit_clips.unsqueeze(-2).expand(-1, -1, N, -1)              ## [S, C, N, N]
        zero_pad = torch.zeros([S, C, N, 1], device=video_feats.device)         
        end_clips = torch.cat((end_clips, zero_pad), dim=-1)                    ## [S, C, N, N + 1]
        end_clips = end_clips[:, :, :, 1:]                                      ## [S, C, N, N]
        
        ## start
        start_concated = torch.cat([video_feats, start_clips, sent_feats], dim=1)
        start_offset = self.start_offset_head(start_concated)                   ## [S, 1, N, N]
        
        ## end 
        end_concated = torch.cat([video_feats, end_clips, sent_feats], dim=1)
        end_offset = self.end_offset_head(end_concated)                         ## [S, 1, N, N]
            
        return start_offset, end_offset

