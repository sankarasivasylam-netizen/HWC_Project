#!/usr/bin/env python
# coding: utf-8

# ### CNN V7 Stage 2

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
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torchvision.models import densenet121, DenseNet121_Weights
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# In[3]:


#ßprint(sys.executable)


# In[8]:




# ### Custom waste dataset to read from metadata

# In[24]:


# Define custom Dataset -> this will help you load images from your csv file

class CustomWasteData(Dataset):
    def __init__(self, csv_df, split,label_col, transform=None):
        self.df = csv_df
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        
        self.transform = transform
        self.label_col = label_col
        
        self.classes = sorted(self.df[label_col].unique())
        self.class_to_idx = {
            cls: i for i, cls in enumerate(self.classes)
        }

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img = Image.open(row["filepath"]).convert("RGBA").convert("RGB")    
        label = self.class_to_idx[row[self.label_col]]
        source = row["source"]     
        type_image = row["type"]    

        if self.transform:
            img = self.transform(img)

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


class SE_Block(torch.nn.Module):
    def __init__(self, c, r=8):
        super(SE_Block, self).__init__()
        self.squeeze = torch.nn.AdaptiveAvgPool2d(1)
        self.excitation = torch.nn.Sequential(
            torch.nn.Conv2d(c, c // r, kernel_size=1,bias=False),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(c // r, c, kernel_size= 1,bias=False),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        y = self.squeeze(x)         
        y = self.excitation(y)    

        return x * y   


class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride=1,use_se=False):
        super(ResidualBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)
        
        self.use_se = use_se
        if use_se:
            self.se = SE_Block(out_channels)
        
        self.activation = torch.nn.ReLU(inplace=True)
        
        self.shortcut = torch.nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        #if self.use_se:
        #    out = self.se(out)
        out += self.shortcut(x)
        out = self.activation(out)
        return out


# In[ ]:


class CNN_stage1_res(torch.nn.Module):
    def __init__(self, num_classes=13):
        super(CNN_stage1_res, self).__init__()
        self.in_channels = 32
        #self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        #self.bn1 = torch.nn.BatchNorm2d(32)
        #self.relu = torch.nn.LeakyReLU(0.1)
        
        self.initialconv = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, padding=1,bias=False),
            torch.nn.BatchNorm2d(32),
            torch.nn.LeakyReLU(0.1, inplace=True)
        )
        
        self.layer1 = self._make_layer(ResidualBlock, 64, 2, stride=2)
        self.layer2 = self._make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(ResidualBlock, 256, 1, stride=2)
        self.layer4 = self._make_layer(ResidualBlock, 512, 1, stride=2)

        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        # Dense layers
        self.fc1 = torch.nn.Linear(512, 1024)
        self.dropout = torch.nn.Dropout(0.3)
        self.fc2 = torch.nn.Linear(1024, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride,use_se=False):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride,use_se))
            self.in_channels = out_channels
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        F = torch.nn.functional
        out = self.initialconv(x)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = self.avgpool(out)
        out = torch.flatten(out, 1)

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

def train_model(model,train_loader_s2,val_loader_s2,device,output_dir):

        num_epochs = 2
        learning_rate = 0.0001
        weight_decay = 0.01
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=2
        )
        early_stopping = EarlyStopping(patience=10, delta=0.001)

        # Training loop
        epoch_times = []
        train_acc_list = []
        train_loss_list = []
        val_acc_list = []
        val_loss_list= []
        best_val_acc = 0.0
        best_epoch = 0
        
        model_name = model.__class__.__name__
        
        for epoch in range(num_epochs):
            start = time.time()
            model.train()
            running_loss = 0.0
            train_loss = 0.0
            train_accuracy = 0.0
            all_predictions = []
            all_labels = []
            for images, labels, sources, type_images in train_loader_s2:
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            # Average loss and accuracy
            train_loss = running_loss / len(train_loader_s2)
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
            
            model.eval()
            all_predictions = []
            all_labels = []
            running_loss = 0.0
            val_loss = 0.0
            val_accuracy = 0.0
            with torch.no_grad():
                for images, labels, sources, type_images in val_loader_s2:
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    running_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    all_predictions.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            val_loss = running_loss / len(val_loader_s2)
            val_accuracy = 100 * sum(np.array(all_predictions) == np.array(all_labels)) / len(all_labels)
            
            #if val_accuracy > best_val_acc + 0.001 :
            #    best_val_acc = val_accuracy
            #    best_epoch = epoch
                
            #    filename_best = f"best_{model_name}_{timestamp}.pth"
                
            #    torch.save({
            #        "model_state_dict": model.state_dict(),
            #        "optimizer_state_dict": optimizer.state_dict(),
            #         "epoch": epoch,
            #        'val_acc': val_accuracy
            #     }, output_dir/filename_best)

            val_loss_list.append(val_loss)
            val_acc_list.append(val_accuracy)

            print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}')
            print(f'Epoch [{epoch+1}/{num_epochs}], Validation Accuracy: {val_accuracy:.4f}')

            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                 num_epochs = epoch + 1
                 print("Early stopping")
                 break
            print('-----------------------------')

        early_stopping.load_best_model(model)
        total_train_time = sum(epoch_times)
        print(f"Total training time: {total_train_time/60:.2f} minutes")

        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total number of parameters: {total_params}")

        filename = f"{model_name}_{timestamp}.pth"
        
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
         }, output_dir/filename)


        # In[171]:


        epochs = range(1, num_epochs +1)
        acc_title = f"Training-Validation_Accuracy_{model_name}"
        plt.title(acc_title)
        plt.plot(epochs,train_acc_list, label="Train Acc")
        plt.plot(epochs,val_acc_list, label="Val Acc")
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.xticks(epochs)
        plt.ylim(0, 100)
        plt.legend()
        acc_fig = f"Training-Validation_Accuracy_{model_name}_{timestamp}.png"
        plt.savefig(os.path.join(output_dir, acc_fig))
        plt.close()


        # In[172]:

        loss_title = f"Training-Validation_Loss_{model_name}"
        plt.title(loss_title)
        plt.plot(epochs,train_loss_list, label="Train loss")
        plt.plot(epochs,val_loss_list, label="Val loss")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.xticks(epochs)
        plt.legend()
        loss_fig = f"Training-Validation_Loss_{model_name}_{timestamp}.png"
        plt.savefig(os.path.join(output_dir, loss_fig))
        plt.close()
        
        return total_train_time #,filename_best


        # In[174]:

def eval_model(model,val_loader_s2,device,output_dir,class_names, train_time):
        
        # Validation
        #print(best_model_file)
       # ckpt = torch.load(output_dir/best_model_file, map_location=device)
        #model.load_state_dict(ckpt["model_state_dict"])
        
        model.eval()
        all_predictions = []
        all_labels = []
        start = time.time()
        with torch.no_grad():
            for images, labels, sources, type_images in val_loader_s2:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        end = time.time()
        num_images= len(val_loader_s2.dataset)
        avg_time = (end - start) / num_images
        throughput = num_images / (end - start)


        # Calculate metrics
        accuracy = 100 * sum(np.array(all_predictions) == np.array(all_labels)) / len(all_labels)
        precision = precision_score(all_labels, all_predictions, average='weighted')
        recall = recall_score(all_labels, all_predictions, average='weighted')
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        balanced_acccuracy=100 * balanced_accuracy_score(all_labels, all_predictions)
        mcc = matthews_corrcoef(all_labels, all_predictions)

        model_name = model.__class__.__name__

        images, labels, sources, type_images = next(iter(val_loader_s2))

        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(images)

            probs = torch.softmax(outputs, dim=1)
            confs, preds = torch.max(probs, dim=1)

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

        img_val_filename = f"image_validation_{model_name}_{timestamp}.png"

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, img_val_filename))
        plt.close()


        # In[186]:


        cm = confusion_matrix(all_labels, all_predictions,)
        fig, ax= plt.subplots()
        sns.heatmap(cm, annot=True, fmt='g', ax=ax,xticklabels=class_names,yticklabels=class_names);
 
        # labels, title and ticks
        cm_title = f"Confusion Matrix {model_name}"
        ax.set_xlabel('Predicted labels');
        ax.set_ylabel('True labels'); 
        ax.set_title(cm_title); 

        cm_filename = f"confusion_matrix_{model_name}_{timestamp}.png"
        
        fig.savefig(os.path.join(output_dir, cm_filename))
        plt.close()

        cr_filename = f"classification_report_{model_name}_{timestamp}.csv"
        
        report = classification_report(all_labels, all_predictions,output_dict=True,
                                       target_names=class_names)
        #with open(os.path.join(output_dir, cr_filename), "w") as f:
        #    f.write(report)

        report_df = pd.DataFrame(report).transpose()

        report_df_classes = report_df.iloc[:len(class_names)]
        report_df_agg = report_df.iloc[len(class_names):]

        macro_f1 = report_df_agg.loc["macro avg", "f1-score"]
        
        report_df_classes.to_csv(output_dir/cr_filename)
        
        
        total_params = sum(p.numel() for p in model.parameters())
        
        eval_metrics_df = pd.DataFrame({
        "params" : [round(total_params/1000000,2)], 
        "trainingtime" : [round(train_time/60,2)],
        "accuracy": [round(accuracy,2)],
        "precision": [round(precision * 100,2)],
        "recall": [round(recall * 100,2)],
        "f1_score": [round(f1 * 100,2)],
        "balanced_accuracy" : [round(balanced_acccuracy,2)],
        "MCC": [round(mcc,2)],
        "Inference time per image" : [round(avg_time*1000,2)],
        "Throughput" : [round(throughput,2)],
        "Macro F1" : [round(macro_f1,2)]
        })

        eval_metrics_file = f"eval_metrics_{model_name}_{timestamp}.csv"
        eval_metrics_df.to_csv(os.path.join(output_dir, eval_metrics_file), index=False)

def main():

        print("CNN Stage2 V5")

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
        file1=META_DIR/"consolidated_metatdata-stg2.csv"
        output_dir = BASE_DIR / "output" / "Stage2" / "cnn_custom" / "V6"


        # In[10]:
        metadata_df = pd.read_csv(file1)

        stg2_meta_df = metadata_df.loc[~metadata_df['stage2_label'].isin(['Non_Organic','Organic'])].copy()


        # In[11]:


        train_df, temp_df = train_test_split(
            stg2_meta_df,
            test_size=0.30,          # 70 train / 30 temp
            stratify=stg2_meta_df["stage2_label"],
            random_state=42
        )

        val_df, test_df = train_test_split(
            temp_df,
            test_size=0.50,          # 15 / 15
            stratify=temp_df["stage2_label"],
            random_state=42
        )


        # In[12]:


        stg2_meta_df.loc[stg2_meta_df["filepath"].isin(train_df["filepath"]), "split"] = "train"
        stg2_meta_df.loc[stg2_meta_df["filepath"].isin(test_df["filepath"]), "split"] = "test"
        stg2_meta_df.loc[stg2_meta_df["filepath"].isin(val_df["filepath"]), "split"] = "val"


        # ### Define transforms

        # In[22]:


        train_transform = transforms.Compose([
            transforms.Resize((192, 192)),
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
            ])

        # In[23]:


        val_transform = transforms.Compose([
            transforms.Resize((192, 192)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
            ])

        # ### Data Loaders

        # In[25]:


        # Creating datasets - Stage 1 
        train_ds_stage2 = CustomWasteData(
            stg2_meta_df,
            split="train",
            label_col="stage2_label",
            transform=train_transform
        )

        val_ds_stage2 = CustomWasteData(
            stg2_meta_df,
            split="val",
            label_col="stage2_label",
            transform=val_transform
        )



        # In[28]:

        train_loader_s2 = DataLoader(
            train_ds_stage2,
            batch_size=64,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            persistent_workers=False
        )

        val_loader_s2 = DataLoader(
            val_ds_stage2,
            batch_size=64,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=False
        )

        # In[169]:

        
        class_names = val_ds_stage2.classes
        
        # In[169]:
        print('-------------------------------------')
        print("CNN Model eval and training - Stage 2"  )
        print('-------------------------------------')
        CNN_stage1_res_model = CNN_stage1_res(13).to(device)
        train_time = train_model(CNN_stage1_res_model,train_loader_s2,val_loader_s2,device,output_dir)
        eval_model(CNN_stage1_res_model,val_loader_s2,device,output_dir,class_names,train_time)
        
        # In[170]:


if __name__ == "__main__":
    main()
