"""
Prediction Script for Node.js Server
Loads model and makes predictions
"""

import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel, ViTConfig
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================

class AttentionMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128], output_dim=128, dropout=0.3):
        super(AttentionMLP, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        self.input_norm = nn.LayerNorm(hidden_dims[0])
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dims[0], num_heads=8, dropout=dropout, batch_first=True
        )
        layers = []
        for i in range(len(hidden_dims) - 1):
            layers.extend([
                nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                nn.LayerNorm(hidden_dims[i + 1]),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
        self.mlp_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)
        self.output_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.input_layer(x)
        x = self.input_norm(x)
        x = F.gelu(x)
        x_attn = x.unsqueeze(1)
        attn_output, _ = self.attention(x_attn, x_attn, x_attn)
        attn_output = attn_output.squeeze(1)
        x = x + self.dropout(attn_output)
        x = self.mlp_layers(x)
        x = self.output_layer(x)
        x = self.output_norm(x)
        return x

class GatedMultimodalFusion(nn.Module):
    def __init__(self, image_dim, tabular_dim, fusion_dim=256, dropout=0.3):
        super(GatedMultimodalFusion, self).__init__()
        self.image_projection = nn.Sequential(
            nn.Linear(image_dim, fusion_dim), nn.LayerNorm(fusion_dim),
            nn.GELU(), nn.Dropout(dropout)
        )
        self.tabular_projection = nn.Sequential(
            nn.Linear(tabular_dim, fusion_dim), nn.LayerNorm(fusion_dim),
            nn.GELU(), nn.Dropout(dropout)
        )
        self.gate_image = nn.Sequential(nn.Linear(fusion_dim, fusion_dim), nn.Sigmoid())
        self.gate_tabular = nn.Sequential(nn.Linear(fusion_dim, fusion_dim), nn.Sigmoid())
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim, num_heads=8, dropout=dropout, batch_first=True
        )
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim), nn.LayerNorm(fusion_dim),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim), nn.LayerNorm(fusion_dim)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, image_features, tabular_features):
        img_proj = self.image_projection(image_features)
        tab_proj = self.tabular_projection(tabular_features)
        img_gate = self.gate_image(img_proj)
        tab_gate = self.gate_tabular(tab_proj)
        img_gated = img_proj * img_gate
        tab_gated = tab_proj * tab_gate
        img_attended, _ = self.cross_attention(
            img_gated.unsqueeze(1), tab_gated.unsqueeze(1), tab_gated.unsqueeze(1)
        )
        img_attended = img_attended.squeeze(1)
        tab_attended, _ = self.cross_attention(
            tab_gated.unsqueeze(1), img_gated.unsqueeze(1), img_gated.unsqueeze(1)
        )
        tab_attended = tab_attended.squeeze(1)
        combined = torch.cat([img_attended, tab_attended], dim=1)
        fused = self.fusion_layer(combined)
        fused = fused + img_gated + tab_gated
        return fused

class AirPollutionMultimodalModel(nn.Module):
    def __init__(self, num_classes, num_tabular_features, vit_model_name='google/vit-base-patch16-224',
                 tabular_hidden_dims=[256, 128], fusion_dim=256, dropout=0.3, pretrained=False):
        super(AirPollutionMultimodalModel, self).__init__()
        self.num_classes = num_classes
        if pretrained:
            self.vit = ViTModel.from_pretrained(vit_model_name)
        else:
            config = ViTConfig.from_pretrained(vit_model_name)
            self.vit = ViTModel(config)
        self.vit_output_dim = self.vit.config.hidden_size
        self.tabular_mlp = AttentionMLP(
            input_dim=num_tabular_features, hidden_dims=tabular_hidden_dims,
            output_dim=128, dropout=dropout
        )
        self.fusion = GatedMultimodalFusion(
            image_dim=self.vit_output_dim, tabular_dim=128,
            fusion_dim=fusion_dim, dropout=dropout
        )
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2), nn.LayerNorm(fusion_dim // 2),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, num_classes)
        )
    
    def forward(self, image, tabular):
        vit_outputs = self.vit(pixel_values=image)
        image_features = vit_outputs.last_hidden_state[:, 0]
        tabular_features = self.tabular_mlp(tabular)
        fused_features = self.fusion(image_features, tabular_features)
        logits = self.classifier(fused_features)
        return logits

# =============================================================================
# PREPROCESSING
# =============================================================================

def get_transform():
    return A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = get_transform()
    augmented = transform(image=image)
    image_tensor = augmented['image'].unsqueeze(0)
    return image_tensor

def preprocess_tabular(params):
    features = [
        params['Year'], params['AQI'], params['PM25'], params['PM10'],
        params['O3'], params['CO'], params['SO2'], params['NO2']
    ]
    hour = params['Hour']
    month = params['Month']
    day = params['Day']
    features.extend([
        np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
        np.sin(2 * np.pi * month / 12), np.cos(2 * np.pi * month / 12),
        np.sin(2 * np.pi * day / 31), np.cos(2 * np.pi * day / 31)
    ])
    features = np.array(features, dtype=np.float32)
    # Simple standardization
    means = np.array([2023.5, 150.0, 75.0, 100.0, 40.0, 1.0, 10.0, 30.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    stds = np.array([2.0, 100.0, 50.0, 80.0, 30.0, 2.0, 20.0, 40.0, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7])
    scaled = (features - means) / (stds + 1e-8)
    return torch.FloatTensor(scaled).unsqueeze(0)

# =============================================================================
# MAIN PREDICTION
# =============================================================================

def main():
    try:
        # Get arguments
        if len(sys.argv) != 4:
            raise ValueError("Usage: python predict.py <model_path> <image_path> <params_json>")
        
        model_path = sys.argv[1]
        image_path = sys.argv[2]
        params = json.loads(sys.argv[3])
        
        # Configuration
        NUM_CLASSES = 6
        NUM_TABULAR_FEATURES = 14
        CLASS_NAMES = ['a_Good', 'b_Satisfactory', 'c_Moderate', 'd_Poor', 'e_VeryPoor', 'f_Severe']
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load model
        model = AirPollutionMultimodalModel(
            num_classes=NUM_CLASSES,
            num_tabular_features=NUM_TABULAR_FEATURES,
            pretrained=False
        )
        
        checkpoint = torch.load(model_path, map_location=DEVICE)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(DEVICE)
        model.eval()
        
        # Preprocess inputs
        image_tensor = preprocess_image(image_path).to(DEVICE)
        tabular_tensor = preprocess_tabular(params).to(DEVICE)
        
        # Predict
        with torch.no_grad():
            outputs = model(image_tensor, tabular_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            pred_class = outputs.argmax(dim=1).item()
            confidence = probabilities[0, pred_class].item()
        
        # Prepare result
        result = {
            'prediction': pred_class,
            'class_name': CLASS_NAMES[pred_class],
            'confidence': confidence,
            'probabilities': {i: probabilities[0, i].item() for i in range(NUM_CLASSES)},
            'input_params': params
        }
        
        # Output JSON
        print(json.dumps(result))
        sys.exit(0)
        
    except Exception as e:
        error_result = {'error': str(e)}
        print(json.dumps(error_result), file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
