import torch
import torch.nn as nn
import timm

class FaceStatusModel(nn.Module):
    def __init__(self, base_model_name="tf_efficientnet_b0_ns", num_classes=4):
        super().__init__()
        
        self.base_model = timm.create_model(base_model_name, pretrained=False)
        self.base_model.classifier = nn.Identity()
        
        feature_dim = self.base_model.num_features
        
        self.relu = nn.ReLU()
        self.fc_au = nn.Linear(feature_dim, feature_dim)
        
        self.dropout = nn.Dropout(p=0.5)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        features = self.base_model(x)
        features = self.relu(self.fc_au(features))
        
        features = self.dropout(features)
        
        logits = self.classifier(features)
        return logits