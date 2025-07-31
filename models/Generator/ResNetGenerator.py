import torch.nn as nn


class ResnetBlock(nn.Module):
  def __init__(self, channels: int):
    super().__init__()
    self.block = nn.Sequential(
       #反射填充+卷积+实例归一化+ReLU
       nn.ReflectionPad2d(1),
       nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=0, bias=False),
       nn.InstanceNorm2d(channels, affine=True, track_running_stats=False),
       #再次反射填充+卷积+实例归一化+ReLU
       nn.ReflectionPad2d(1),
       nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=0, bias=False),
       nn.InstanceNorm2d(channels, affine=True, track_running_stats=False),
    )

  def forward(self, x):
    #残差连接
    return x + self.block(x)
  

class ResNetGenerator(nn.Module):
  def __init__(self, in_channels=3, out_channels=3, n_residual_blocks=9):
    super().__init__()
    layers = []
    #先对输入的图像做一个大小为7×7的卷积、一个InstanceNorm2d、一个ReLU
    layers += [
        nn.ReflectionPad2d(3),
        nn.Conv2d(in_channels, 64, kernel_size=7, stride=1, padding=0, bias=False),
        nn.InstanceNorm2d(64, affine=True, track_running_stats=False),
        nn.ReLU(inplace=True)
    ]
    #对图像做两次下采样
    in_c = 64
    out_c = 128
    for _ in range(2):
      layers += [
          nn.Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1, bias=False),
          nn.InstanceNorm2d(out_c, affine=True, track_running_stats=False),
          nn.ReLU(inplace=True)
      ]
      in_c = out_c
      out_c *= 2
    #9层的ResnetBlock
    for _ in range(n_residual_blocks):
      layers += [ResnetBlock(in_c)]
    #两次双线性插值的3×3的上采样
    for _ in range(2):
      layers += [
          nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
          nn.Conv2d(in_c, in_c//2, kernel_size=3, stride=1, padding=1, bias=False),
          nn.InstanceNorm2d(in_c//2, affine=True, track_running_stats=False),
          nn.ReLU(inplace=True)
      ]
      in_c = in_c//2
    #最后采用7x7的卷积输出图像
    layers += [
        nn.ReflectionPad2d(3),
        nn.Conv2d(64, out_channels, kernel_size=7, stride=1, padding=0, bias=False),
        nn.Tanh()
    ]
    self.model = nn.Sequential(*layers)

  def forward(self,x):
    return self.model(x)