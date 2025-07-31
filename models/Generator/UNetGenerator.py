import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetGenerator(nn.Module):
  def __init__(self, in_channels=3, out_channels=3, features=[64,128,256,512]):
    super().__init__()
    #下采样包括两次3×3的卷积、两次InstanceNorm2d、两次ReLU
    self.downs = nn.ModuleList()
    for feature in features:
      self.downs.append(
        nn.Sequential(
          nn.Conv2d(in_channels, feature, kernel_size=3, padding=1, bias=False),
          nn.InstanceNorm2d(feature),
          nn.ReLU(inplace=True),
          nn.Conv2d(feature, feature, kernel_size=3, padding=1, bias=False),
          nn.InstanceNorm2d(feature),
          nn.ReLU(inplace=True)
        )
      )
      in_channels = feature
    self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    #bottleneck
    self.bottleneck = nn.Sequential(
      nn.Conv2d(features[-1], features[-1]*2, kernel_size=3, padding=1, bias=False),
      nn.InstanceNorm2d(features[-1]*2),
      nn.ReLU(inplace=True),
      nn.Conv2d(features[-1]*2, features[-1]*2, kernel_size=3, padding=1, bias=False),
      nn.InstanceNorm2d(features[-1]*2),
      nn.ReLU(inplace=True)
    )
    #上采用采用2倍的双线性插值、InstanceNorm2d，跳跃连接之后再进行3×3的卷积
    self.ups = nn.ModuleList()
    for feature in reversed(features):
      self.ups.append(
        nn.Sequential(
          nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
          nn.Conv2d(feature*3, feature, kernel_size=3, padding=1, bias=False),
          nn.InstanceNorm2d(feature),
          nn.ReLU(inplace=True),
          nn.Conv2d(feature, feature, kernel_size=3, padding=1, bias=False),
          nn.InstanceNorm2d(feature),
          nn.ReLU(inplace=True)
        )
      )
    #最后进行1×1的卷积
    self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1, padding=0)

  def forward(self, x):
    skip_connections = []
    for down in self.downs:
      x = down(x)
      skip_connections.append(x)
      x = self.pool(x)
    x = self.bottleneck(x)
    skip_connections = skip_connections[::-1]  #先取出最深层的跳跃连接与上采样特征拼接，再逐层向浅层对齐恢复
    for idx, up in enumerate(self.ups):
      x = up[0](x)
      skip = skip_connections[idx]
      #若尺寸不相同，采用插值使其对齐
      if x.shape[2:] != skip.shape[2:]:
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
      x = torch.cat((skip, x), dim=1) #concat
      x = up[1:](x)

    return torch.tanh(self.final_conv(x))