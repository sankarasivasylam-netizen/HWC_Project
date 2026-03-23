#!/usr/bin/env python
# coding: utf-8

# ### Custom CNN - Version 3 - Deep model

# In[38]:


from pathlib import Path
import pandas as pd
import os
import shutil
import hashlib
import seaborn as sns
from IPython.display import display
import time
import sys
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.models import resnet34, ResNet34_Weights
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import matthews_corrcoef
import random


# In[3]:


print(sys.executable)


# In[8]:




# ### Custom waste dataset to read from metadata

# In[24]:


# Define custom Dataset -> this will help you load images from your csv file

class CustomWasteData(Dataset):
    def __init__(self, csv_df, split,label_col, transforms_dict,train=True):
        self.df = csv_df
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        
        #self.transform = transform
        self.label_col = label_col
        self.transforms_dict = transforms_dict
        self.train = train
        
        self.classes = sorted(self.df[label_col].unique())
        self.class_to_idx = {
            cls: i for i, cls in enumerate(self.classes)
        }

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img = Image.open(row["filepath"]).convert("RGB")
        label = self.class_to_idx[row[self.label_col]]
        source = row["source"]     
        type_image = row["type"]    

        if self.train:
            if row["stage1_label"] == "Organic":
               if row["dup_flag"] == 1:
                   img = self.transforms_dict["org_dup"](img)
               else:
                   img = self.transforms_dict["org_base"](img)

            else:
                img = self.transforms_dict["non_organic"](img)
        else:
            img = self.transforms_dict["default"](img)

        return img, label, source, type_image


# In[ ]:


class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict()
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.counter = 0

    def load_best_model(self, model):
        model.load_state_dict(self.best_model_state)


# In[ ]:


class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)
        self.activation = torch.nn.LeakyReLU(0.1)
        
        self.shortcut = torch.nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.activation(out)
        return out


# In[ ]:


class CNN_stage1_res(torch.nn.Module):
    def __init__(self, num_classes=2):
        super(CNN_stage1_res, self).__init__()
        self.in_channels = 32
        #self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        #self.bn1 = torch.nn.BatchNorm2d(32)
        #self.relu = torch.nn.LeakyReLU(0.1)
        
        self.initialconv = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, padding=1,bias=False),
            torch.nn.BatchNorm2d(32),
            torch.nn.LeakyReLU(0.1)
        )
        
        self.layer1 = self._make_layer(ResidualBlock, 64, 1, stride=1)
        self.layer2 = self._make_layer(ResidualBlock, 128, 1, stride=2)
        self.layer3 = self._make_layer(ResidualBlock, 128, 1, stride=2)
        self.layer4 = self._make_layer(ResidualBlock, 256, 1, stride=2)
        self.layer5 = self._make_layer(ResidualBlock, 256, 1, stride=2)

        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        # Dense layers
        self.fc1 = torch.nn.Linear(256, 512)
        self.dropout = torch.nn.Dropout(0.3)
        self.fc2 = torch.nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        F = torch.nn.functional
        out = self.initialconv(x)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)

        out = self.fc1(out)
        out = F.leaky_relu(out,0.1)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

def denormalize(img):
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])

        img = img.cpu().numpy().transpose(1, 2, 0)
        img = std * img + mean
        img = np.clip(img, 0, 1)

        return img

def hide_and_seek(image, **kwargs):  
    """
    Less aggressive Hide-and-Seek occlusion technique without altering colors.
    """
    if isinstance(image, np.ndarray):
        img = np.transpose(image, (2, 0, 1))
        img = img.copy()  
    else:
        img = image.clone() if isinstance(image, torch.Tensor) else image  
 
    c, h, w = img.shape  
    grid_sizes = [12, 24, 36, 48]
    hide_prob = 0.2  # Reduced probability to erase smaller portions
 
    grid_size = random.choice(grid_sizes)
 
    for x in range(0, w, grid_size):
        for y in range(0, h, grid_size):
            x_end = min(w, x + grid_size)
            y_end = min(h, y + grid_size)
            if random.random() <= hide_prob:
                img[:, y:y_end, x:x_end] = 0
 
    if isinstance(image, np.ndarray):
        img = np.transpose(img, (1, 2, 0))  
 
    return img

class hide_seek_transform:
    def __call__(self, img):
        img_np = np.array(img)
        img_np = hide_and_seek(img_np)  # your original function
        return Image.fromarray(img_np)

def main():

        print("CNN test run 3")
        
        os.getcwd()
        os.chdir("..")
        os.getcwd()

        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        if device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        print("Using device:", device)

        # In[9]:


        BASE_DIR = Path.cwd()
        META_DIR = BASE_DIR / "data" / "metadata"
        file1=META_DIR/"consolidated_metatdata.csv"
        output_dir = BASE_DIR / "output"


        # In[10]:


        metadata_df = pd.read_csv(file1)


        # In[11]:


        train_df, temp_df = train_test_split(
            metadata_df,
            test_size=0.30,          # 70 train / 30 temp
            stratify=metadata_df["stage2_label"],
            random_state=42
        )

        val_df, test_df = train_test_split(
            temp_df,
            test_size=0.50,          # 15 / 15
            stratify=temp_df["stage2_label"],
            random_state=42
        )


        # In[12]:


        metadata_df.loc[metadata_df["filepath"].isin(train_df["filepath"]), "split"] = "train"
        metadata_df.loc[metadata_df["filepath"].isin(test_df["filepath"]), "split"] = "test"
        metadata_df.loc[metadata_df["filepath"].isin(val_df["filepath"]), "split"] = "val"


        # In[16]:


        val_df = metadata_df.loc[metadata_df['split'] == "val"]
        val_df["dup_flag"] = 0

        test_df = metadata_df.loc[metadata_df['split'] == "test"]
        test_df["dup_flag"] = 0

        organic_df = metadata_df.loc[ (metadata_df['stage1_label'] == "Organic") & (metadata_df['split'] == "train") ]
        organic_df["dup_flag"] = 0

        # Duplicate
        organic_dup = organic_df.copy()
        organic_dup["dup_flag"] = 1

        nonorganic_df = metadata_df.loc[(metadata_df['stage1_label'] == "Non_Organic") & (metadata_df['split'] == "train")]
        nonorganic_df["dup_flag"] = 0


        # In[17]:


        updated_metadata_df = pd.concat(
            [organic_df, organic_dup, nonorganic_df,val_df, test_df],
            ignore_index=True
        ).sample(frac=1)


        # In[19]:


        updated_metadata_df['stage1_label'].value_counts()


        # ### Define transforms

        # In[22]:
        
        organic_standard = transforms.Compose([
            transforms.RandomResizedCrop(150, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4
            ),
            transforms.ToTensor(),
        ])

        organic_erasing = transforms.Compose([
            transforms.Resize((150, 150)),
            transforms.ToTensor(),
            transforms.RandomErasing(
                p=1.0,
                scale=(0.05, 0.2),
                ratio=(0.3, 3.3),
                value=0
                ),
            ])

        organic_hide_seek = transforms.Compose([
            transforms.Resize((150, 150)),
            hide_seek_transform(),
            transforms.ToTensor(),
            ])


        train_transforms = {
            "non_organic": transforms.Compose([
            transforms.Resize((150,150)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
            ]), 
            "org_base":transforms.Compose([
            transforms.Resize((150,150)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
            ]),
            "org_dup":transforms.Compose([
            transforms.RandomChoice([
                organic_standard,
                organic_erasing,
                organic_hide_seek
            ]),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
            ])
        }


        # In[23]:


        val_transform = {
            
            "default": transforms.Compose([
            transforms.Resize((150, 150)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
           ])
        }

        # ### Data Loaders

        # In[25]:


        # Creating datasets - Stage 1 
        train_ds_stage1 = CustomWasteData(
            updated_metadata_df,
            split="train",
            label_col="stage1_label",
            transforms_dict=train_transforms,
            train=True
        )

        val_ds_stage1 = CustomWasteData(
            updated_metadata_df,
            split="val",
            label_col="stage1_label",
            transforms_dict=val_transform,
            train=False
        )


        # In[28]:


        from torch.utils.data import DataLoader

        train_loader_s1 = DataLoader(
            train_ds_stage1,
            batch_size=128,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            persistent_workers=False
        )

        val_loader_s1 = DataLoader(
            val_ds_stage1,
            batch_size=128,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            persistent_workers=False
        )


        # In[169]:

        
        CNN_stage1_res_model = CNN_stage1_res(2).to(device)

        num_epochs = 1
        learning_rate = 0.0001
        weight_decay = 0.01
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            CNN_stage1_res_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=1
        )
        early_stopping = EarlyStopping(patience=10, delta=0.001)


        # In[170]:


        # Training loop
        epoch_times = []
        train_acc_list = []
        train_loss_list = []
        val_acc_list = []
        val_loss_list= []
        for epoch in range(num_epochs):
            start = time.time()
            CNN_stage1_res_model.train()
            running_loss = 0.0
            train_loss = 0.0
            train_accuracy = 0.0
            all_predictions = []
            all_labels = []
            for images, labels, sources, type_images in train_loader_s1:
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = CNN_stage1_res_model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            # Average loss and accuracy
            train_loss = running_loss / len(train_loader_s1)
            train_accuracy = 100 * sum(np.array(all_predictions) == np.array(all_labels)) / len(all_labels)

            train_loss_list.append(train_loss)
            train_acc_list.append(train_accuracy)
                
            end = time.time()
            epoch_time = end - start
            epoch_times.append(epoch_time)

            print(f"Epoch {epoch+1} time: {epoch_time:.2f} sec")
            print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss:.4f}')
            print(f'Epoch [{epoch+1}/{num_epochs}], Training Accuracy: {train_accuracy:.4f}')

            
            
            scheduler.step() # for LR scheduler
            
            CNN_stage1_res_model.eval()
            all_predictions = []
            all_labels = []
            running_loss = 0.0
            val_loss = 0.0
            val_accuracy = 0.0
            with torch.no_grad():
                for images, labels, sources, type_images in val_loader_s1:
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = CNN_stage1_res_model(images)
                    loss = criterion(outputs, labels)
                    running_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    all_predictions.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            val_loss = running_loss / len(val_loader_s1)
            val_accuracy = 100 * sum(np.array(all_predictions) == np.array(all_labels)) / len(all_labels)

            val_loss_list.append(val_loss)
            val_acc_list.append(val_accuracy)

            print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}')
            print(f'Epoch [{epoch+1}/{num_epochs}], Validation Accuracy: {val_accuracy:.4f}')

            early_stopping(val_loss, CNN_stage1_res_model)
            if early_stopping.early_stop:
                num_epochs = epoch + 1
                print("Early stopping")
                break
            print('-----------------------------')

        early_stopping.load_best_model(CNN_stage1_res_model)
        total_time = sum(epoch_times)
        print(f"Total training time: {total_time/60:.2f} minutes")
        
        total_params = sum(p.numel() for p in CNN_stage1_res_model.parameters())
        print(f"Total number of parameters: {total_params}")
        
        torch.save({
            "model_state_dict": CNN_stage1_res_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
         }, output_dir/"cnn_res_3.pth")


        # In[171]:

        model_name = CNN_stage1_res_model.__class__.__name__


        epochs = range(1, num_epochs +1)
        acc_title = f"Training-Validation_Accuracy_{model_name}"
        plt.title(acc_title)
        plt.plot(epochs,train_acc_list, label="Train Acc")
        plt.plot(epochs,val_acc_list, label="Val Acc")
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.xticks(range(0, len(train_acc_list), 5))
        plt.ylim(0, 100)
        plt.legend()
        acc_fig = f"Training-Validation_Accuracy_{model_name}_.png"
        plt.savefig(os.path.join(output_dir, acc_fig))
        plt.close()


        # In[172]:

        loss_title = f"Training-Validation_Loss_{model_name}"
        plt.title(loss_title)
        plt.plot(epochs,train_loss_list, label="Train loss")
        plt.plot(epochs,val_loss_list, label="Val loss")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.xticks(range(0, len(train_loss_list), 5))
        plt.legend()
        loss_fig = f"Training-Validation_Loss_{model_name}_.png"
        plt.savefig(os.path.join(output_dir, loss_fig))
        plt.close()



        # In[174]:


        # Validation
        CNN_stage1_res_model.eval()
        all_predictions = []
        all_labels = []
        start = time.time()
        with torch.no_grad():
            for images, labels, sources, type_images in val_loader_s1:
                images = images.to(device)
                labels = labels.to(device)
                outputs = CNN_stage1_res_model(images)
                _, predicted = torch.max(outputs.data, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        end = time.time()
        num_images= len(val_loader_s1.dataset)
        avg_time = (end - start) / num_images
        throughput = num_images / (end - start)


        # Calculate metrics
        accuracy = 100 * sum(np.array(all_predictions) == np.array(all_labels)) / len(all_labels)
        precision = precision_score(all_labels, all_predictions, average='weighted')
        recall = recall_score(all_labels, all_predictions, average='weighted')
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        balanced_acccuracy=100 * balanced_accuracy_score(all_labels, all_predictions)
        mcc = matthews_corrcoef(all_labels, all_predictions)


        eval_metrics_df = pd.DataFrame({
        "accuracy": [accuracy],
        "precision": [precision],
        "recall": [recall],
        "f1_score": [f1],
        "balanced_accuracy" : [balanced_acccuracy],
        "MCC": [mcc],
        "Inference time per image" : [avg_time*1000],
        "Throughput" : [throughput],
        "Total training time" : [total_time/60]
        })

        eval_metrics_file = f"eval_metrics_{model_name}.csv"
        eval_metrics_df.to_csv(os.path.join(output_dir, eval_metrics_file), index=False)

            
        images, labels, sources, type_images = next(iter(val_loader_s1))

        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = CNN_stage1_res_model(images)

            probs = torch.softmax(outputs, dim=1)
            confs, preds = torch.max(probs, dim=1)

        class_names = val_ds_stage1.classes

        fig, axes = plt.subplots(3, 6, figsize=(12, 6))

        for i, ax in enumerate(axes.flatten()):

            img = denormalize(images[i])
            true_label = class_names[labels[i]]
            pred_label = class_names[preds[i]]
            conf = confs[i].item()

            src = sources[i]
            type_image = type_images[i]

            ax.imshow(img)
            ax.set_title(
                f"T: {true_label}\nP: {pred_label} ({conf:.2f})\n"
                f"{src} | {type_image}",fontsize=8
            )
            ax.axis("off")

        img_val_filename = f"image_validation_{model_name}.png"

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, img_val_filename))
        plt.close()


        # In[186]:


        cm = confusion_matrix(all_labels, all_predictions, labels=[1,0])
        fig, ax= plt.subplots()
        sns.heatmap(cm, annot=True, fmt='g',ax=ax);
 
        # labels, title and ticks
        cm_title = f"Confusion Matrix {model_name}"
        ax.set_xlabel('Predicted labels');
        ax.set_ylabel('True labels'); 
        ax.set_title(cm_title); 
        ax.xaxis.set_ticklabels(['Organic', 'Non-Organic']);
        ax.yaxis.set_ticklabels(['Organic', 'Non-Organic'])

        cm_filename = f"confusion_matrix_{model_name}.png"
        
        fig.savefig(os.path.join(output_dir, cm_filename))
        plt.close()

        cr_filename = f"classification_report_{model_name}.txt"
        
        report = classification_report(all_labels, all_predictions)
        with open(os.path.join(output_dir, cr_filename), "w") as f:
            f.write(report)


if __name__ == "__main__":
    main()
