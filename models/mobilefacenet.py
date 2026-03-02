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

class Bottleneck(nn.Module):
    """Khối Residual Bottleneck như trong kiến trúc MobileNetV2."""
    def __init__(self, in_c, out_c, stride, t):
        super().__init__()
        # Chỉ dùng kết nối thặng dư (residual connection) khi stride=1 và số channel không đổi
        self.use_res_connect = stride == 1 and in_c == out_c
        exp_c = in_c * t # Hệ số mở rộng t
        
        self.conv = nn.Sequential(
            # 1. Pointwise (Mở rộng số lượng channel)
            ConvBlock(in_c, exp_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0)),
            # 2. Depthwise (Tích chập theo từng channel riêng biệt)
            ConvBlock(exp_c, exp_c, kernel=(3, 3), stride=(stride, stride), padding=(1, 1), groups=exp_c),
            # 3. Linear Pointwise (Chiếu lại về số channel mong muốn, không dùng PReLU)
            LinearBlock(exp_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        )
        
    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)