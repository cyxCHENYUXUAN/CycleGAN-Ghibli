import torch
import torch.nn as nn


class TransformerDecoderGenerator(nn.Module):
  def __init__(self, in_channels=3, out_channels=3, feature_dim=256, num_heads=8, num_layers=6, img_size=128, patch_size=4):
    super().__init__()
    self.patch_embed = nn.Conv2d(in_channels, feature_dim, kernel_size=4, stride=4)
    #位置编码
    num_patches = (img_size//patch_size)**2
    self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, feature_dim))
    #解码器
    decoder_layer = nn.TransformerDecoderLayer(d_model=feature_dim, nhead=num_heads, batch_first=True, norm_first=True)
    self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
    #卷积+双线性插值
    self.reconstruct = nn.Sequential(
      nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
      nn.Conv2d(feature_dim, out_channels, kernel_size=3, padding=1),
      nn.Tanh()
    )

  def forward(self, x):
    feat = self.patch_embed(x)
    B,D,Hf,Wf = feat.shape
    #加入位置编码
    tokens = feat.flatten(2).transpose(1,2)
    tokens = tokens + self.pos_embed
    #此处解码器同时将原始tokens作为输入和上下文
    decoder_out = self.transformer_decoder(tokens, tokens)
    recon = decoder_out.transpose(1,2).reshape(B,D,Hf,Wf)
    return self.reconstruct(recon)