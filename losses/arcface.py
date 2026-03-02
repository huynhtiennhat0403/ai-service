import torch
import torch.nn as nn
import torch.nn.functional as F

class ArcFaceLoss(nn.Module):
    def __init__(self, in_features, out_features, s=64.0, m=0.5):
        """
        Khởi tạo hàm loss ArcFace.
        - in_features: Số chiều của vector đặc trưng (Embedding size, VD: 128 với MobileFaceNet).
        - out_features: Số lượng class (Số lượng ID người trong tập training, VD: 10571).
        """
        super(ArcFaceLoss, self).__init__()
        self.s = s
        self.m = m
        # Khởi tạo ma trận trọng số W có kích thước (out_features, in_features)
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)