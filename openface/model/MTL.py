import torch
import torch.nn as nn
import timm

from .AU_model import *

class View_MTL(nn.Module):
    def __init__(self, base_model_name='tf_efficientnet_b0_ns', pretrained: bool = False, pos_dim: int = 4):
        super(View_MTL, self).__init__()
        # Keep backbone separate so it can be frozen during training
        self.base_model = timm.create_model(base_model_name, pretrained=pretrained)
        self.base_model.classifier = nn.Identity()
        
        feature_dim = self.base_model.num_features
        self.pos_dim = pos_dim

        self.relu = nn.ReLU()

        self.fc_view = nn.Linear(feature_dim, feature_dim)
        self.fc_pos = nn.Linear(pos_dim, feature_dim // 4)
        
        self.view_regressor = nn.Linear(feature_dim + feature_dim // 4, 2)  

    def forward(self, x, pos=None):
        features = self.base_model(x)

        if pos is None:
            pos = torch.zeros(x.size(0), self.pos_dim, device=x.device, dtype=features.dtype)

        features_view = self.relu(self.fc_view(features))
        features_pos = self.relu(self.fc_pos(pos))
        fused = torch.cat([features_view, features_pos], dim=1)
        view_output = self.view_regressor(fused)
        
        return view_output

class MTL(nn.Module):
    def __init__(self, base_model_name='tf_efficientnet_b0_ns', expr_classes=8, au_numbers=8):
        super(MTL, self).__init__()
        self.base_model = timm.create_model(base_model_name, pretrained=False)
        self.base_model.classifier = nn.Identity()
        
        feature_dim = self.base_model.num_features

        self.relu = nn.ReLU()

        self.fc_emotion = nn.Linear(feature_dim, feature_dim)
        self.fc_gaze = nn.Linear(feature_dim, feature_dim)
        self.fc_au = nn.Linear(feature_dim, feature_dim)
        
        self.emotion_classifier = nn.Linear(feature_dim, expr_classes)
        self.gaze_regressor = nn.Linear(feature_dim, 2)  
        # self.au_regressor = nn.Linear(feature_dim, au_numbers)  
        self.au_regressor = Head(in_channels=feature_dim, num_classes=au_numbers, neighbor_num=4, metric='dots')

    def forward(self, x):
        features = self.base_model(x)

        features_emotion = self.relu(self.fc_emotion(features))
        features_gaze = self.relu(self.fc_gaze(features))
        features_au = self.relu(self.fc_au(features))
        
        emotion_output = self.emotion_classifier(features_emotion)
        gaze_output = self.gaze_regressor(features_gaze)
        # au_output = torch.sigmoid(self.au_regressor(features_au))
        au_output = self.au_regressor(features_au)
        
        return emotion_output, gaze_output, au_output
