import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    """Lớp tích chập cơ bản đi kèm Batch Normalization và hàm kích hoạt PReLU."""
    def __init__(self, in_c, out_c, kernel=(1,1), stride=(1,1), padding=(0,0), groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_c, out_channels=out_c,kernel_size=kernel,stride=stride, padding=padding, groups=groups, bias=False)
        self.bn= nn.BatchNorm2d(out_c)
        self.prelu = nn.PReLU(out_c)

    def forward(self, x):
        return self.prelu(self.bn(self.conv(x)))

class LinearBlock(nn.Module):
    """Lớp 'Linear' không có hàm kích hoạt phi tuyến tính ở cuối."""
    def __init__(self, in_c, out_c, kernel=(1,1), stride=(1,1), padding=(0,0), groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
    
    def forward(self, x):
        return self.bn(self.conv(x))
