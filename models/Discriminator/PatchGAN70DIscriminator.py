import torch.nn as nn

class Discriminator(nn.Module):
  def __init__(self, in_channels=3):
    super().__init__()
    layers = []
    #先对输入的图像做一个4×4和stride=2的卷积、一个LeakyReLU
    layers += [
        nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1, bias=False),
        nn.LeakyReLU(0.2, inplace=True)
    ]
    #对图像做三次4×4的卷积（前两个stride=2，最后一个stride=1）、三次InstanceNorm2d、三次LeakyReLU
    in_c = 64
    out_c = 128
    for i in range(3):
      stride = 2 if i < 2 else 1
      layers += [
          nn.Conv2d(in_c, out_c, kernel_size=4, stride=stride, padding=1, bias=False),
          nn.InstanceNorm2d(out_c, affine=True, track_running_stats=False),
          nn.LeakyReLU(0.2, inplace=True)
      ]
      in_c = out_c
      out_c *= 2
    #最后一层采用4×4的卷积，stride=1，然后输出featureMap
    layers += [
        nn.Conv2d(in_c, 1, kernel_size=4, stride=1, padding=1, bias=False)
    ]
    self.model = nn.Sequential(*layers)

  def forward(self, x):
    return self.model(x)