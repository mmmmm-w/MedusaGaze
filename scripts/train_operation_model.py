import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import pandas as pd
import glob
from tqdm import tqdm

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from openface.operation_model import FaceStatusModel

CONFIG = {
    "data_root": "./data/OperationData_Crops",
    "mtl_weights": "./weights/MTL_backbone.pth",
    "save_dir": "./weights/trained_models",
    "batch_size": 16,
    "lr": 5e-5,
    "epochs": 20,
    "img_size": 224,
    "num_classes": 4
}

class FaceOperationDataset(Dataset):
    def __init__(self, root_dir, transform=None, mode='all', val_split=0.2):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        csv_files = glob.glob(os.path.join(root_dir, "*", "labels.csv"))
        
        if not csv_files:
            raise RuntimeError(f"No labels.csv files found in {root_dir}.")

        print(f"[{mode.upper()}] Scanning {len(csv_files)} session files...")

        all_data = []

        for csv_file in csv_files:
            session_dir = os.path.dirname(csv_file)
            img_dir = os.path.join(session_dir, "images")
            
            df = pd.read_csv(csv_file)
            
            if 'timestamp' not in df.columns:
                df['timestamp'] = range(len(df))

            for _, row in df.iterrows():
                img_name = row['filename']
                class_id = int(row['class_id'])
                ts = float(row['timestamp'])
                
                full_img_path = os.path.join(img_dir, img_name)
                
                if os.path.exists(full_img_path):
                    all_data.append({
                        'path': full_img_path,
                        'label': class_id,
                        'time': ts
                    })

        all_data.sort(key=lambda x: x['time'])

        class_map = {}
        for item in all_data:
            c = item['label']
            if c not in class_map:
                class_map[c] = []
            class_map[c].append(item)

        final_samples = []
        for c, items in class_map.items():
            split_idx = int(len(items) * (1 - val_split))
            
            if mode == 'train':
                final_samples.extend(items[:split_idx])
            elif mode == 'val':
                final_samples.extend(items[split_idx:])
            else:
                final_samples.extend(items)

        self.samples = [(x['path'], x['label']) for x in final_samples]
        print(f"[{mode.upper()}] Loaded {len(self.samples)} images.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)

def load_mtl_weights(model, mtl_weights_path, device):
    if not os.path.exists(mtl_weights_path):
        print(f"Warning: MTL weights not found at {mtl_weights_path}. Training from scratch.")
        return model
        
    mtl_state = torch.load(mtl_weights_path, map_location=device)
    model_state = model.state_dict()
    
    pretrained_dict = {
        k: v for k, v in mtl_state.items() 
        if k in model_state and v.shape == model_state[k].shape
    }
    
    model_state.update(pretrained_dict)
    model.load_state_dict(model_state)
    print(f"Loaded {len(pretrained_dict)} layers from MTL checkpoint.")
    return model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs(CONFIG["save_dir"], exist_ok=True)

    train_transforms = transforms.Compose([
        transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = FaceOperationDataset(
        root_dir=CONFIG["data_root"], 
        transform=train_transforms, 
        mode='train'
    )
    
    val_dataset = FaceOperationDataset(
        root_dir=CONFIG["data_root"], 
        transform=val_transforms, 
        mode='val'
    )
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=4)

    model = FaceStatusModel(base_model_name="tf_efficientnet_b0_ns", num_classes=CONFIG["num_classes"])
    model.to(device)
    
    model = load_mtl_weights(model, CONFIG["mtl_weights"], device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"])

    best_acc = 0.0
    
    for epoch in range(CONFIG["epochs"]):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        print(f"\nEpoch {epoch+1}/{CONFIG['epochs']}")
        
        pbar = tqdm(train_loader, desc="Training")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
            
        train_acc = 100 * correct / total
        avg_loss = running_loss / len(train_loader)
        
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total if val_total > 0 else 0
        print(f"Result: Train Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join(CONFIG["save_dir"], "best_operation_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Best model saved to {save_path}")

    torch.save(model.state_dict(), os.path.join(CONFIG["save_dir"], "last_operation_model.pth"))
    print("Training finished.")

if __name__ == "__main__":
    main()