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

class MobileFaceNet(nn.Module):
    def __init__(self, embedding_size=128):
        super().__init__()
        # Theo đúng thông số của Table 1 trong bài báo
        
        # [Input] 112x112x3 -> [Output] 56x56x64
        self.conv1 = ConvBlock(3, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1)) 
        
        # [Input] 56x56x64 -> [Output] 56x56x64 (Depthwise conv3x3)
        self.dw_conv1 = ConvBlock(64, 64, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=64)
        
        # Các khối Bottleneck chính
        self.blocks = nn.Sequential(
            # t=2, c=64, n=5, s=2
            self._make_layer(Bottleneck, t=2, in_c=64, out_c=64, n=5, s=2),
            # t=4, c=128, n=1, s=2
            self._make_layer(Bottleneck, t=4, in_c=64, out_c=128, n=1, s=2),
            # t=2, c=128, n=6, s=1
            self._make_layer(Bottleneck, t=2, in_c=128, out_c=128, n=6, s=1),
            # t=4, c=128, n=1, s=2
            self._make_layer(Bottleneck, t=4, in_c=128, out_c=128, n=1, s=2),
            # t=2, c=128, n=2, s=1
            self._make_layer(Bottleneck, t=2, in_c=128, out_c=128, n=2, s=1)
        )
        
        # [Input] 7x7x128 -> [Output] 7x7x512
        self.conv2 = ConvBlock(128, 512, kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        
        # Linear GDConv7x7
        self.linear_gdconv = LinearBlock(512, 512, kernel=(7, 7), stride=(1, 1), padding=(0, 0), groups=512)
        
        # Linear conv1x1
        self.linear_conv = LinearBlock(512, embedding_size, kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        
    def _make_layer(self, block, t, in_c, out_c, n, s):
        layers = []
        # Lớp đầu tiên của chuỗi sử dụng stride = s
        layers.append(block(in_c, out_c, s, t))
        # Các lớp sau trong chuỗi sử dụng stride = 1
        for i in range(1, n):
            layers.append(block(out_c, out_c, 1, t))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.dw_conv1(out)
        out = self.blocks(out)
        out = self.conv2(out)
        out = self.linear_gdconv(out)
        out = self.linear_conv(out)
        
        # Làm phẳng ma trận thành vector đặc trưng (Batch_size, 128)
        return out.view(out.shape[0], -1)