import torch
import torch.nn.functional as F
from typing import Optional
from torch import nn, Tensor
from learning.components.mlp import MLP
from learning.net.multihead_attention import MultiheadAttention


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, input_channel=3, num_pos_feats=256):
        super().__init__()
        self.position_embedding_head = nn.Sequential(
            nn.Conv1d(input_channel, num_pos_feats, kernel_size=1),
            nn.BatchNorm1d(num_pos_feats),
            nn.ReLU(inplace=True),
            nn.Conv1d(num_pos_feats, num_pos_feats, kernel_size=1))

    def forward(self, xyz):
        # xyz : BxNx3
        xyz = xyz.transpose(1, 2).contiguous()
        # Bx3xN
        position_embedding = self.position_embedding_head(xyz)
        return position_embedding


class Transformer(nn.Module):
    def __init__(self,
                 input_channels=3,
                 use_xyz=True,
                 input_size=4000,
                 d_model=64,
                 num_layers=1,
                 num_heads=1,
                 key_feature_dim=128,
                 with_pos_embed=True,
                 encoder_pos_embed_input_dim=3,
                 fea_channels=(128, 256, 256),
                 feat_slim_last_layer=True,
                 ):
        super(Transformer, self).__init__()
        self.input_channels = input_channels
        self.use_xyz = use_xyz
        self.input_size = input_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.encoder_pos_embed_input_dim = encoder_pos_embed_input_dim
        assert encoder_pos_embed_input_dim == 3
        self.with_pos_embed = with_pos_embed

        assert d_model % num_heads == 0
        assert num_heads == 1, 'Only support num_heads=1'
        multihead_attn = MultiheadAttention(
            feature_dim=d_model, n_head=num_heads, key_feature_dim=key_feature_dim)

        if self.with_pos_embed:
            encoder_pos_embed = PositionEmbeddingLearned(encoder_pos_embed_input_dim, d_model)
        else:
            encoder_pos_embed = None

        self.fea_layer = MLP(fea_channels, batch_norm=True, last_layer=feat_slim_last_layer)
        self.encoder = TransformerEncoder(
            multihead_attn=multihead_attn, FFN=None,
            d_model=d_model, num_encoder_layers=num_layers,
            self_posembed=encoder_pos_embed)

    def encode(self, feature, coord):
        """Use transformer to encode features

            feature : (B, C, N)
            coord : (B, N, 3) or (B, N, 6)
        """
        # BxCxN -> NxBxC
        feature = feature.permute(2, 0, 1)

        # encoder (self-attention)
        encoded_feat = self.encoder(feature,
                                    query_pos=coord if self.with_pos_embed else None)  # (N, B, C)

        encoded_feat = encoded_feat.permute(1, 0, 2)  # NxBxC -> BxNxC
        encoded_feat = self.fea_layer(encoded_feat)  # BxNxC

        return encoded_feat

    def forward(self, feature, coord):
        """
            feature: (B*N, C)
            coord: (B*N, 3)
        """
        feature_size = feature.shape[-1]
        feature = feature.reshape(-1, self.input_size, feature_size).permute(0, 2, 1)  # (B, C, N)
        coord = coord.reshape(-1, self.input_size, 3)  # (B, N, 3)

        att_feature = self.encode(feature, coord)  # (B, N, C)
        return att_feature


class InstanceL2Norm(nn.Module):
    """Instance L2 normalization.
    """
    def __init__(self, size_average=True, eps=1e-5, scale=1.0):
        super().__init__()
        self.size_average = size_average
        self.eps = eps
        self.scale = scale

    def forward(self, input):
        if self.size_average:
            return input * (self.scale * ((input.shape[1] * input.shape[2] * input.shape[3]) / (
                        torch.sum((input * input).reshape(input.shape[0], 1, 1, -1), dim=3, keepdim=True) + self.eps)).sqrt())  # view
        else:
            return input * (self.scale / (torch.sum((input * input).reshape(input.shape[0], 1, 1, -1), dim=3, keepdim=True) + self.eps).sqrt())


class TransformerEncoderLayer(nn.Module):
    def __init__(self, multihead_attn, FFN, d_model, self_posembed=None):
        super().__init__()
        self.self_attn = multihead_attn
        # Implementation of Feedforward model
        self.FFN = FFN
        self.norm = nn.InstanceNorm1d(d_model)
        self.self_posembed = self_posembed

        self.dropout = nn.Dropout(0.1)

    def with_pos_embed(self, tensor, pos_embed: Optional[Tensor]):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, src, query_pos=None):
        # BxNxC -> BxCxN -> NxBxC
        if self.self_posembed is not None and query_pos is not None:
            query_pos_embed = self.self_posembed(query_pos).permute(2, 0, 1)
        else:
            query_pos_embed = None
        query = key = value = self.with_pos_embed(src, query_pos_embed)

        # self-attention
        # NxBxC
        src2 = self.self_attn(query=query, key=key, value=value)
        # TODO: support num_heads > 1
        src = src + src2

        # NxBxC -> BxCxN -> NxBxC
        src = self.norm(src.permute(1, 2, 0)).permute(2, 0, 1)
        return F.relu(src)
        # return src


class TransformerEncoder(nn.Module):
    def __init__(self, multihead_attn, FFN,
                 d_model=512,
                 num_encoder_layers=6,
                 activation="relu",
                 self_posembed=None):
        super().__init__()
        encoder_layer = TransformerEncoderLayer(
            multihead_attn, FFN, d_model, self_posembed=self_posembed)
        self.layers = _get_clones(encoder_layer, num_encoder_layers)

    def forward(self, src, query_pos=None):
        output = src

        for layer in self.layers:
            output = layer(output, query_pos=query_pos)
        return output


def _get_clones(module, N):
    return nn.ModuleList([module for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
