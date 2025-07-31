import torch
import torch.nn as nn


class ConvBlock(nn.Module): #UNet++的基本块
  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.conv = nn.Sequential(
      nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
      nn.InstanceNorm2d(out_channels),
      nn.ReLU(inplace=True),
      nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
      nn.InstanceNorm2d(out_channels),
      nn.ReLU(inplace=True)
    )

  def forward(self, x):
    return self.conv(x)
  

class UNetPPGenerator(nn.Module): #UNet++生成器使用密集连接的跳跃连接和嵌套结构
  def __init__(self, in_channels=3, out_channels=3, features=[64,128,256,512]):
    super().__init__()
    self.nb_filter = features+[features[-1]*2]
    self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    self.nodes = nn.ModuleList()    #存储所有节点

    #构建UNet++的嵌套密集连接结构
    #j表示纵向深度，i表示横向位置
    for j in range(5):
      column_nodes = nn.ModuleList()
      for i in range(5-j):
        out_c = self.nb_filter[i]
        if j==0:
          in_c = in_channels if i==0 else self.nb_filter[i-1]
        else:
          in_c = self.nb_filter[i]*j+self.nb_filter[i+1]
        column_nodes.append(ConvBlock(in_c, out_c))
      self.nodes.append(column_nodes)

    self.final_conv = nn.Conv2d(self.nb_filter[0], out_channels, kernel_size=1)

  def forward(self,x):
    #存储所有节点的输出，x_行号_列号
    node_outputs = {}


    for j in range(5):  #j表示UNet++中从最浅层到最深层的第j层
      for i in range(5-j):  #i表示在第j层中，从左到右第i个节点
        if j==0:    #第一行的节点
          if i==0:    #第一列的节点
            node_input = x
          else:
            pre_output = node_outputs[f'x_{i-1}_{j}']
            node_input = self.pool(pre_output)
        else:
          skip_connections = [node_outputs[f'x_{i}_{k}'] for k in range(j)]
          down_node_output = node_outputs[f'x_{i+1}_{j-1}']
          upsample_input = self.up(down_node_output)
          node_input = torch.cat(skip_connections+[upsample_input], dim=1)

        output = self.nodes[j][i](node_input)
        node_outputs[f'x_{i}_{j}'] = output

    final_output = self.final_conv(node_outputs['x_0_4'])
    return torch.tanh(final_output)