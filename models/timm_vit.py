import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from thop import profile
import random

class VisionTransformer(nn.Module):
    def __init__(
        self,
        num_layers=12,
        top_k=1,
        prune_ratio=0.0,
        target_prune_ratio=0.5,
        prune_step=0.05,
        score_method="attention",
    ):
        super(VisionTransformer, self).__init__()

        # read pre-trained model
        self.backbone = timm.create_model("vit_small_patch16_224", pretrained=True)

        # manually patchify
        self.backbone.patch_embed.proj = nn.Conv2d(1, 384, kernel_size=16, stride=16)

        in_features = self.backbone.head.in_features
        self.backbone.head = nn.Identity()

        self.num_layers = num_layers

        self.transformer_layers = nn.ModuleList(
            [self.backbone.blocks[i] for i in range(self.num_layers)]
        )
        
        self.fc1 = nn.Linear(in_features, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 2)
        
        self.top_k = top_k
        self.score_method = score_method
        self.prune_ratio = prune_ratio
        self.prune_step = prune_step
        self.target_prune_ratio = target_prune_ratio

        self.attention_scores = None
        self.backbone.blocks = None

    def hook_fn(self, module, input, output):
        # see timm/models/vision_transformer
        self.attention_scores = module.attn_drop(output)
        # print("Attention scores shape:", self.attention_scores.shape)

    # store the activation of attention score
    def register_hooks(self):
        for block in self.transformer_layers:
            block.attn.register_forward_hook(self.hook_fn)

    # we didn't apply head pruning in ieeevr
    def prune_heads(self):
        current_prune_ratio = min(self.prune_ratio, self.target_prune_ratio)
        for block in self.transformer_layers:
            attn_weights = self.attention_scores 
            importance_scores = attn_weights.mean(dim=1).mean(dim=1).cpu().numpy()
            num_heads_to_prune = int(block.attn.num_heads * current_prune_ratio)
            pruned_heads = importance_scores.argsort()[:num_heads_to_prune]
            for head in pruned_heads:
                block.attn.head_mask[head] = 0
        
        self.prune_ratio += self.prune_step

    def random_prune_heads(self):
        for block in self.transformer_layers:
            num_heads = block.attn.num_heads
            num_heads_to_prune = num_heads // 3 
            pruned_heads = random.sample(range(num_heads), num_heads_to_prune)

            for head in pruned_heads:
                self.attention_weights[:, head, :, :] = 0.0

    def forward(self, x):
        self.register_hooks()

        x = self.backbone.patch_embed(x)
        if self.backbone.pos_embed.shape[1] == 197 and x.shape[1] == 196:
            pos_embed = self.backbone.pos_embed[:, 1:, :] 
        else:
            pos_embed = self.backbone.pos_embed
    
        x = self.backbone.pos_drop(x + pos_embed)

        for i, block in enumerate(self.transformer_layers):
            # print(block)
            x = block(x)
            if i%2 ==1:
                if self.score_method == "attention":
                    # average value among different features
                    attn_scores = self.attention_scores.mean(dim=-1)
                    
                    topk_indices = attn_scores.topk(int(self.top_k * attn_scores.size(1)), dim=1, largest=True).indices
                    if topk_indices.max() >= x.size(1):
                        raise ValueError("topk_indices contains out of bounds index")
    
                    bs = x.size(0)
                    batch_indices = (
                        torch.arange(bs)
                        .unsqueeze(-1)
                        .expand(-1, topk_indices.size(1))
                        .to(x.device)
                    )
    
                    informative_tokens = x[batch_indices, topk_indices]
    
                    non_informative_indices = torch.ones_like(attn_scores, dtype=bool)
                    non_informative_indices[batch_indices, topk_indices] = False
                    non_informative_tokens = x[non_informative_indices].view(
                        bs, -1, x.size(-1)
                    )
                    x = informative_tokens
                    if non_informative_tokens.size(1) > 0:
                        non_informative_scores = attn_scores[non_informative_indices].view(
                            bs, -1
                        )
                        weighted_sum = (
                            non_informative_tokens * non_informative_scores.unsqueeze(-1)
                        ).sum(dim=1)
                        sum_scores = non_informative_scores.sum(dim=1).unsqueeze(-1)
                        # sum_scores = non_informative_scores.sum(dim=1).unsqueeze(-1)
                        sum_scores = torch.clamp(sum_scores, min=1e-5)  # Clamping to avoid zero values

                        package_token = weighted_sum / (sum_scores+1e-5)
                        x = torch.cat(
                            [informative_tokens, package_token.unsqueeze(1)], dim=1
                        )
                    else:
                        x = informative_tokens

        features = x.mean(dim=1)
        gaze_dir = F.relu(self.fc1(features))
        gaze_dir = F.relu(self.fc2(gaze_dir))
        gaze_dir = F.relu(self.fc3(gaze_dir))
        gaze_dir = self.fc4(gaze_dir)

        return gaze_dir
        
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vit = VisionTransformer(score_method="attention", top_k=0.8, num_layers=6).to(
        device
    )
    print(vit)

    input_image = torch.randn(1, 1, 224, 224).to(device)
    output = vit(input_image)
    print(output.shape)

    flops, params = profile(vit, inputs=(input_image,))

    print(f"Total Params: {params}")
    print(f"Total FLOPs: {flops}")

