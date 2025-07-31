import torch
import torch.nn.functional as F
import torch.nn as nn

class MultiScaleEdgeDiscriminator(nn.Module):
  def __init__(self, in_channels=3, scales=[1, 0.5, 0.25]):
    super().__init__()
    self.scales = scales
    self.discriminators = nn.ModuleList()

    for _ in scales:
      layers = []
      layers += [
        #RGB三通道+边缘图通道输入
        nn.utils.spectral_norm(nn.Conv2d(in_channels + 1, 64, kernel_size=4, stride=2, padding=1)),
        nn.LeakyReLU(0.2, inplace=True)
      ]

      #中间层为3次下采样和通道增加
      in_ch, out_ch = 64, 128
      for _ in range(2):
        layers += [
            nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False)),
            #明确InstanceNorm2d的参数，更适合GAN训练
            nn.InstanceNorm2d(out_ch, affine=True, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        in_ch = out_ch
        out_ch *= 2
      layers += [
        nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=1, padding=1, bias=False)),
        nn.InstanceNorm2d(out_ch, affine=True, track_running_stats=False),
        nn.LeakyReLU(0.2, inplace=True)
      ]
      layers += [nn.utils.spectral_norm(nn.Conv2d(out_ch, 1, kernel_size=3, stride=1, padding=1))]#尺寸不变

      self.discriminators.append(nn.Sequential(*layers))

  def forward(self, x):
    edge = self._get_edge_map(x)
    x_cat = torch.cat([x, edge], dim=1)
    outputs = []
    for scale, disc in zip(self.scales, self.discriminators):
      if scale != 1:
        h, w = x_cat.shape[-2:]
        new_h, new_w = int(h * scale), int(w * scale)
        xi = F.interpolate(x_cat, size=(new_h, new_w), mode='bilinear', align_corners=True)
      else:
        xi = x_cat
      outputs.append(disc(xi))
    return outputs

  def _get_edge_map(self, x):
    #detach()可以防止影响生成器的梯度
    with torch.no_grad():
      gray = x.mean(dim=1, keepdim=True)
      #3x3平均池化
      blurred = F.avg_pool2d(gray, kernel_size=3, stride=1, padding=1)
      edge = torch.abs(gray - blurred)
    return edge.detach()