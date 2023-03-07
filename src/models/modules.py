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
        #self.proj = nn.Conv2d(512, 256, 1) # testing
        # torch.nn.init.normal_(self.proj.weight, std=0.2)

        self.convs = nn.ModuleList()
        for layer_idx, layer_count in enumerate(self.num_scale_layers):
            ## first layer
            if layer_idx == 0:
                '''
                self.convs.extend([
                    nn.Sequential(
                        nn.Conv1d(hidden_size, hidden_size, 1, 1),
                        nn.BatchNorm1d(hidden_size),
                        nn.ReLU(),
                    )
                ])
                
                for count in range(1, layer_count):
                    if (count % 2) != 0: ## 1, 3, 5 ...
                        self.convs.extend([nn.MaxPool1d(2, 1)])
                    else: ## 2, 4, 6 ...
                        self.convs.extend([
                            nn.Sequential(
                                nn.Conv1d(hidden_size, hidden_size, 2, 1),
                                nn.BatchNorm1d(hidden_size),
                                nn.ReLU(),
                            )
                        ])
                '''
                '''
                ## maxpool except first input
                for i in range(1, layer_count):
                    self.convs.extend([
                        nn.MaxPool1d(2, 1),
                    ])

                '''
                self.convs.extend([nn.Sequential(
                    nn.Conv1d(hidden_size, hidden_size, 2, 1),
                    nn.BatchNorm1d(hidden_size),
                    nn.ReLU(),
                ) for _ in range(layer_count-1)])  

            ## other layers 
            else: 
                self.convs.extend([nn.MaxPool1d(3, 2)])
                self.convs.extend([nn.Sequential(
                    nn.Conv1d(hidden_size, hidden_size, 2, 1),
                    nn.BatchNorm1d(hidden_size),
                    nn.ReLU(),
                ) for _ in range(layer_count-1)])            

class SparsePropConv(nn.Module):
    def __init__(self, counts, hidden_size):
        super().__init__()
        self.num_scale_layers = counts
        self.hidden_size = hidden_size
        # torch.nn.init.normal_(self.proj.weight, std=0.2)

        self.convs = nn.ModuleList()
        for layer_idx, layer_count in enumerate(self.num_scale_layers):
            ## first layer
            if layer_idx == 0:
                self.convs.extend([nn.Sequential(
                    nn.Conv1d(hidden_size, hidden_size, 2, 1),
                    nn.BatchNorm1d(hidden_size),
                    nn.ReLU(),
                ) for _ in range(layer_count-1)])  

            ## other layers 
            else: 
                self.convs.extend([nn.MaxPool1d(3, 2)])
                self.convs.extend([nn.Sequential(
                    nn.Conv1d(hidden_size, hidden_size, 2, 1),
                    nn.BatchNorm1d(hidden_size),
                    nn.ReLU(),
                ) for _ in range(layer_count-1)])            
      
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
                if accumulate_count != 0:
                    x = self.convs[accumulate_count-1](x)
               
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
        self.offset_predictor = nn.Sequential(
                                    nn.Conv2d(in_channel*2, in_channel, 1),
                                    nn.ReLU(),
                                    nn.Dropout(0.5),
                                    nn.Conv2d(in_channel, 2, 1),
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
        ## channel-wise concat
        concated_feats = torch.cat([video_feats, sent_feats], dim=1)  ## [S, C+C, N, N] 
        offset = self.offset_predictor(concated_feats)     ## [S, 2, N, N],  delta_s and delta_e
        
        return offset

     
class ProposalConv_PE(nn.Module):
    def __init__(
        self,
        in_channel: int,            # input feature size
        hidden_channel: int,        # hidden feature size
        out_channel: int,           # output feature size
        kernel_size: int,           # kernel size
        num_layers: int,            # number of CNN layers (exclude the projection layers)
    ):
        super(ProposalConv_PE, self).__init__()
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
            
            ## init log_sigma_proj to have small weight
            #self.init_uncertainty_module_weight()

    def init_uncertainty_module_weight(self):
        nn.init.xavier_uniform_(self.log_sigma_proj.weight)
        # scale down to prevent too large log_sigma
        self.log_sigma_proj.weight = self.log_sigma_proj.weight * 0.1 
        nn.init.constant_(self.log_sigma_proj.bias, 0)


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

            # clamp log_sigma
            ## log_sigma = 1.15 -> log_variance = 2.3 -> variance = 10
            ## log_sigma = -1.15 -> log_variance = -2.3 -> variance = 0.1
            x_log_sigma = torch.clamp(x_log_sigma, min=-1.15, max=1.15) 

        return x_mean, x_log_sigma, mask2d

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

        ## init log_sigma_proj to have small weight
        #self.init_uncertainty_module_weight()

    def init_uncertainty_module_weight(self):
        nn.init.xavier_uniform_(self.log_sigma_proj[1].weight)
        # scale down to prevent too large log_sigma
        self.log_sigma_proj.weight = self.log_sigma_proj[1].weight * 0.1 
        nn.init.constant_(self.log_sigma_proj[1].bias, 0)

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

        # clamp log_sigma
        ## log_sigma = 1.15 -> log_variance = 2.3 -> variance = 10
        ## log_sigma = -1.15 -> log_variance = -2.3 -> variance = 0.1
        log_sigma_feats = torch.clamp(log_sigma_feats, min=-1.15, max=1.15)
        
        return mean_feats, log_sigma_feats
