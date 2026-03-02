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
    
    def forward(self, embedding, label):
        # Bước 1 & 2: Chuẩn hóa L2 cho feature đầu vào (x) và trọng số (W)
        embedding_norm = F.normalize(embedding, p=2, dim=1)
        weight_norm = F.normalize(self.weight, p=2, dim=1)

        # Bước 3 & 4: Tính cos(theta) thông qua tích vô hướng (Tương đương FullyConnected không bias)
        # original_target_logit
        cosine = F.linear(embedding_norm, weight_norm)

        # Tránh lỗi NaN khi tính arccos do sai số thập phân khiến giá trị vượt ra ngoài [-1, 1]
        cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        # Bước 5: Tính góc theta
        theta = torch.acos(cosine)

        # Bước 6: Cộng thêm margin m vào góc theta và tính lại cos
        # marginal_target_logit = cos(theta + m)
        marginal_target_logit = torch.cos(theta + self.m)

        # Bước 7: Tạo One-hot vector cho ground truth label
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1.0)

        # Bước 8: Áp dụng margin m chèn vào vị trí ground truth (những vị trí không phải ground truth giữ nguyên cosine)
        logits = (one_hot * marginal_target_logit) + ((1.0 - one_hot) * cosine)

        # Bước 9: Nhân logits với hệ số scale s
        logits = logits * self.s

        # Đầu ra (Class-wise affinity score) sẽ được đưa vào hàm CrossEntropyLoss tiêu chuẩn sau đó
        return logits