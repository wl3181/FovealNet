import torch
import torch.nn as nn
import numpy as np
from torchsummary import summary
from sklearn.cluster import MiniBatchKMeans
from transformers import ViTModel, ViTConfig
import pickle
from thop import profile


class PatchEmbeddings(nn.Module):
    def __init__(self, d_model: int, patch_size: int, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, d_model, patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        bs, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, num_patches: int):
        super(PositionalEncoding, self).__init__()
        self.positional_encoding = nn.Parameter(torch.randn(1, num_patches, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) < self.positional_encoding.size(1):
            return x + self.positional_encoding[:, : x.size(1), :]
        elif x.size(1) > self.positional_encoding.size(1):
            padding = self.positional_encoding[:, -1:, :].expand(
                1, x.size(1) - self.positional_encoding.size(1), -1
            )
            return x + torch.cat([self.positional_encoding, padding], dim=1)
        else:
            return x + self.positional_encoding


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == embed_dim
        ), "Embedding dimension must be divisible by number of heads"

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.fc_out = nn.Linear(embed_dim, embed_dim)

        self.pruned_heads = set()
        self.attn = None
        self.head_mask = torch.ones(num_heads, requires_grad=False)

    def forward(self, x: torch.Tensor, return_attention=False):
        N, seq_len, _ = x.shape

        qkv = self.qkv(x).reshape(N, seq_len, self.num_heads, 3, self.head_dim)
        qkv = qkv.permute(3, 0, 2, 1, 4)
        queries, keys, values = qkv.chunk(3, dim=0)
        queries = queries.squeeze(0)
        keys = keys.squeeze(0)
        values = values.squeeze(0)

        self.head_mask = self.head_mask.to(queries.device)

        queries = queries * self.head_mask.view(1, -1, 1, 1)
        keys = keys * self.head_mask.view(1, -1, 1, 1)
        values = values * self.head_mask.view(1, -1, 1, 1)

        energy = torch.matmul(queries, keys.transpose(-1, -2))
        attention = torch.softmax(energy / (self.head_dim**0.5), dim=-1)
        self.attn = attention

        out = torch.matmul(attention, values)
        self.attn_out = out
        out = out.transpose(1, 2).reshape(N, seq_len, self.num_heads * self.head_dim)
        out = self.fc_out(out)
        if return_attention:
            return out, attention
        return out

    def compute_head_importance(self):
        with torch.no_grad():
            importance_scores = torch.zeros(self.num_heads)
            for i in range(self.num_heads):
                importance_scores[i] = torch.mean(self.attn_out[:, i, :, :])
            return importance_scores.cpu().numpy()

    def prune_heads(self, prune_ratio):
        importance_scores = self.compute_head_importance()
        num_heads_to_prune = int(self.num_heads * prune_ratio)
        pruned_heads = np.argsort(importance_scores)[:num_heads_to_prune]
        self.pruned_heads.update(pruned_heads)

        for head in pruned_heads:
            self.head_mask[head] = 0


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float = 0.1,
        return_score: bool = False,
        top_k: float = 0.6,
    ):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.return_score = return_score
        self.top_k = top_k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.return_score:
            x = x + self.attention(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
            return x
        else:
            attn_output, attention_scores = self.attention(
                self.norm1(x), return_attention=True
            )
            x = x + attn_output
            x = x + self.mlp(self.norm2(x))

            scores = attention_scores.mean(dim=1).mean(dim=1)
            topk_indices = scores.topk(
                int(self.top_k * x.size(1)), dim=1, largest=False, sorted=False
            ).indices
            bs = x.size(0)
            batch_indices = (
                torch.arange(bs)
                .unsqueeze(-1)
                .expand(-1, topk_indices.size(1))
                .to(x.device)
            )
            informative_tokens = x[batch_indices, topk_indices]

            non_informative_indices = torch.ones_like(scores, dtype=bool)
            non_informative_indices[batch_indices, topk_indices] = False
            non_informative_tokens = x[non_informative_indices].view(bs, -1, x.size(-1))

            if non_informative_tokens.size(1) > 0:
                non_informative_scores = scores[non_informative_indices].view(bs, -1)
                weighted_sum = (
                    non_informative_tokens * non_informative_scores.unsqueeze(-1)
                ).sum(dim=1)
                sum_scores = non_informative_scores.sum(dim=1).unsqueeze(-1)
                package_token = weighted_sum / sum_scores
                x = torch.cat([informative_tokens, package_token.unsqueeze(1)], dim=1)
            else:
                x = informative_tokens

            return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=112,
        patch_size=16,
        in_channels=1,
        num_outputs=2,
        embed_dim=768,  
        num_heads=12, 
        mlp_dim=3072,  
        num_layers=12,  
        dropout=0.1,
        top_k=0.6,
        n_clusters=10,
        score_method="clustering",
        update_interval=100,
    ):
        super(VisionTransformer, self).__init__()
        self.patch_embedding = PatchEmbeddings(embed_dim, patch_size, in_channels)
        num_patches = (img_size // patch_size) ** 2
        self.positional_encoding = PositionalEncoding(embed_dim, num_patches)
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim,
                    num_heads,
                    mlp_dim,
                    dropout,
                    top_k=top_k,
                    return_score=(score_method == "attention"),
                )
                for _ in range(num_layers)
            ]
        )
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(embed_dim)
        self.num_heads = num_heads
        self.regressors = nn.ModuleList(
            [nn.Linear(embed_dim, num_outputs) for _ in range(num_layers)]
        )
        self.top_k = int(top_k * num_patches)
        self.n_clusters = n_clusters
        self.score_method = score_method
        self.update_interval = update_interval
        self.kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=0)
        self.prune_ratio = 0.0
        self.prune_step = 0.05
        self.target_prune_ratio = 0.5

        self.pretrained_model = ViTModel.from_pretrained(
            "google/vit-base-patch16-224-in21k"
        )
        self.load_pretrained_weights()

    def load_pretrained_weights(self):
        pretrained_state_dict = self.pretrained_model.state_dict()

        pretrained_conv_weight = (
            self.pretrained_model.embeddings.patch_embeddings.projection.weight.data
        )
        new_conv_weight = pretrained_conv_weight.mean(dim=1, keepdim=True)
        self.patch_embedding.conv.weight.data = new_conv_weight
        self.patch_embedding.conv.bias.data = (
            self.pretrained_model.embeddings.patch_embeddings.projection.bias.data
        )

        self.positional_encoding.positional_encoding.data = (
            self.pretrained_model.embeddings.position_embeddings.data
        )

        for i, block in enumerate(self.pretrained_model.encoder.layer):
            if i >= len(self.transformer_blocks):
                break

            query_weight = block.attention.attention.query.weight.data
            key_weight = block.attention.attention.key.weight.data
            value_weight = block.attention.attention.value.weight.data

            query_bias = block.attention.attention.query.bias.data
            key_bias = block.attention.attention.key.bias.data
            value_bias = block.attention.attention.value.bias.data

            qkv_weight = torch.cat([query_weight, key_weight, value_weight], dim=0)
            qkv_bias = torch.cat([query_bias, key_bias, value_bias], dim=0)
            assert (
                self.transformer_blocks[i].attention.qkv.weight.data.shape
                == qkv_weight.shape
            )
            assert (
                self.transformer_blocks[i].attention.qkv.bias.data.shape
                == qkv_bias.shape
            )

            self.transformer_blocks[i].attention.qkv.weight.data = qkv_weight
            self.transformer_blocks[i].attention.qkv.bias.data = qkv_bias

            self.transformer_blocks[i].attention.fc_out.weight.data = (
                block.attention.output.dense.weight.data
            )
            self.transformer_blocks[i].attention.fc_out.bias.data = (
                block.attention.output.dense.bias.data
            )

            self.transformer_blocks[i].mlp[
                0
            ].weight.data = block.intermediate.dense.weight.data
            self.transformer_blocks[i].mlp[
                0
            ].bias.data = block.intermediate.dense.bias.data

            self.transformer_blocks[i].mlp[
                2
            ].weight.data = block.output.dense.weight.data
            self.transformer_blocks[i].mlp[2].bias.data = block.output.dense.bias.data

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embedding(x)
        x = self.positional_encoding(x)

        outputs = []
        for i, block in enumerate(self.transformer_blocks):
            x = block(x)
            x_norm = self.norm(x.mean(dim=1))
            outputs.append(self.regressors[i](x_norm))

        return outputs

    def prune_heads(self):
        current_prune_ratio = min(self.prune_ratio, self.target_prune_ratio)
        for layer in self.transformer:
            if hasattr(layer, "attention"):
                layer.attention.prune_heads(current_prune_ratio)
        self.prune_ratio += self.prune_step


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vit = VisionTransformer(score_method="attention", top_k=0.2, num_layers=4).to(
        device
    )
    print(vit)

    input_image = torch.randn(3, 1, 224, 224).to(device)
    output = vit(input_image)
    print(output.shape)

    flops, params = profile(vit, inputs=(input_image,))

    print(f"Total Params: {params}")
    print(f"Total FLOPs: {flops}")
