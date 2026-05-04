#!/usr/bin/env python
# coding: utf-8

#### CNN Stage 1 - Training and Evaluation script

#Import libraries

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
from torchvision.transforms import v2
from torch.utils.data import WeightedRandomSampler
from sklearn.utils.class_weight import compute_class_weight
from torchvision.transforms.v2 import CutMix, RandomChoice, Identity
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
import math
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchinfo import summary

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


#Custom wastedataset for Dataloader

class CustomWasteData(Dataset):
    """
    Custom Pytorch dataset for Stage-1 classification
    
    Loads images from the metadata csv file with file location and label
    Applies transformations for training and validation
    
    Args
    
        csv_df      : Metadata dataframe with file path and labels
        split       : Train, val or test string
        label_col   : To distinguish stage-1 / stage-2 string
        transforms_dict : Torchvision V2 transforms to apply on each image
        train       : train boolean flag used to apply right transformation for train or val
    
    Attributes
    
        df          : Metadata dataframe
        label_col   : Train, val or test
        transforms_dict : Torchvision V2 transforms to apply on each image
        classes     : Sorted list of class names
        class_to_idx : Dict Class name to integer label mapping
        target      :  List of integer labels for all samples
        
    Returns
        PIL Image, integer true label , Str image source, Str image type (clean or real)
        
    
    """
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
        self.targets = [self.class_to_idx[l] for l in self.df[label_col].tolist()]

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img = Image.open(row["filepath"]).convert("RGBA").convert("RGB")
        label = self.class_to_idx[row[self.label_col]]
        source = row["source"]     
        type_image = row["type"]    
        #To apply the right transform for train / val-test
        if self.train:
            img = self.transforms_dict["standard"](img)
        else:
            img = self.transforms_dict["default"](img)

        return img, label, source, type_image


class EarlyStopping:
    """
        Early stopping to stop tranining when validation loss doesnt improve for 
        set patience
        
    """
    
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


class ResidualBlock(torch.nn.Module):
    """
    Custom Residual block for building CNN
    
    Loads images from the metadata csv file with file location and label
    Applies transformations for training and validation
    
    Args
    
        in_channels : number of input channels (int)
        out_channels : number of output channels (int)
        stride      : convolution stride (int)
    
    Attributes
    
        conv1, conv2 :  convolution layers
        bn1, bn2     :  batch normalization
        activation   :  leaky ReLU activation function
        shortcut     :  skip connection
        
        
    Returns
        output tensor of shape batch size x out_channels x h x w

    
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)
        self.activation = torch.nn.LeakyReLU(0.1)
        
        self.shortcut = torch.nn.Sequential()
        #Skip connection when stride is not 1 or when input / output channels dont match
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

class CNN_stage1_res(torch.nn.Module):
    """
    Custom Residual CNN architecture designed as  baseline
    
    The architecture seelected based on Saraçoğlu & Çetin Kaya (2025) but with a reduced depth of 5 residual layers
    simpler for binary classification. It is a purposefully built lightweight residual network trained from scratch 
    as baseline. It consists of an initial convolution block followed by 5 residual stages, global average pooling 
    and the classifier head. Within each stage, skip connection allows direct gradient flow through network during 
    training preventing vanishing gradients issue like ResNet.
    
    Args
    
        num_classes : number of classes : 2 for stage1 (int)
    
    Attributes
    
        conv1, conv2 :  convolution layers
        bn1, bn2     :  batch normalization
        activation   :  leaky ReLU activation function
        shortcut     :  skip connection
    
    """
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
        #5 residual layers each with 1 residual bloack
        self.layer1 = self._make_layer(ResidualBlock, 64, 1, stride=1)
        self.layer2 = self._make_layer(ResidualBlock, 128, 1, stride=2)
        self.layer3 = self._make_layer(ResidualBlock, 256, 1, stride=2)
        self.layer4 = self._make_layer(ResidualBlock, 512, 1, stride=2)
        self.layer5 = self._make_layer(ResidualBlock, 512, 1, stride=2)

        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        # Dense layers
        self.fc1 = torch.nn.Linear(512, 256)
        self.dropout = torch.nn.Dropout(0.3)
        self.fc2 = torch.nn.Linear(256, num_classes)

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
    """
    Denormalize the input image tensor for plotting it without normalization effects
    
    """
    #ImageNet mean and standard deviation used for normalization/denormalization
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    img = img.cpu().numpy().transpose(1, 2, 0)
    img = std * img + mean
    img = np.clip(img, 0, 1)

    return img
		 
def train_model(model,train_loader_s1,val_loader_s1,device,output_dir, train_transforms,val_transform,train_transforms_2,val_transform_2,train_ds_stage1,val_ds_stage1):
        """
        Main training function that executes the full training pipline
        
        1. Training configuration and parameter setup including adaptive learning rate setup 
        2. Progressive resizing and augmentation (cutmix)
        3. Training and validation loops for number of epochs
        4. Training and validation metric computation
        5. Early stopping and best model checkpointing
        6. Training/validation accuracy loss plots
        
        Args:
        
            model              : initialized model
            train_loader_s1    : Train loader for stage1
            val_loader_s1      : Val loader for stage1
            device             : device in which training is running (mps/gou etc)
            output_dir         : direcory to save files
            train_transforms   : Torchvision transform for training data
            val_transform      : Torchvision transform for val data
            train_transforms_2 : Phase 2 train transform
            val_transform_2    : Phase 2 val transform
            train_ds_stage1    : Train dataset
            val_ds_stage1      : val dataset
    
        Returns:
        
            total_train_time : Total training time in seconds
            best_model_file  : Filename of the best model checkpoint
        
        """
        num_epochs = 50
        #Best LR based on hyperparameter tuning
        learning_rate = 0.00004 
        weight_decay = 0.01
		
        model_name = model.__class__.__name__
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay)
	    
	    #ReduceLROnPlateau with min val loss and a patience of 3 to adjust LR
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                                optimizer,
                                mode='min',
                                patience=3,factor=0.5,
                                min_lr=1e-7,
                                threshold=0.001,	
                                threshold_mode='abs',
                                verbose=True
                                )
        early_stopping = EarlyStopping(patience=7, delta=0.001)

        #CutMix application - 30% probability in a batch
        cutmix = CutMix(num_classes=2, alpha=1.0)
        cutmix_or_identity = RandomChoice(
                                          [cutmix, Identity()],
                                          p=[0.3, 0.7]
                                         )

        # Training loop
        epoch_times = []
        train_acc_list = []
        train_loss_list = []
        val_acc_list = []
        val_loss_list= []
        best_val_acc = 0
        best_epoch = 0
        
        for epoch in range(num_epochs):
            start = time.time()
            model.train()
            running_loss = 0.0
            train_loss = 0.0
            train_accuracy = 0.0
            all_predictions = []
            all_labels = []

            

            if epoch == 25:
	    
		        #Phase wise training - Switch transform for progressive resizing
                train_ds_stage1.transforms_dict = train_transforms_2
                val_ds_stage1.transforms_dict = val_transform_2
                print("Switched to phase 3 transform")
                
                print(train_ds_stage1.transforms_dict)
                
                for images, labels, sources, type_images in train_loader_s1:
                    print("Train batch shape:", images.shape)
                    break
		          
						
            for images, labels, sources, type_images in train_loader_s1:
                #CutMix applied after 10 epochs
                if epoch >= 10: 
                    images, labels = cutmix_or_identity(images, labels)
                    
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
                all_labels.extend((labels.argmax(dim=1) if labels.ndim == 2 else labels).cpu().numpy())

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

            
            model.eval()
            all_predictions = []
            all_labels = []
            running_loss = 0.0
            val_loss = 0.0
            val_accuracy = 0.0
            with torch.no_grad():
                for images, labels, sources, type_images in val_loader_s1:
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    running_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    all_predictions.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            val_loss = running_loss / len(val_loader_s1)
            val_accuracy = 100 * sum(np.array(all_predictions) == np.array(all_labels)) / len(all_labels)
            mcc = matthews_corrcoef(all_labels, all_predictions)
			
            scheduler.step(val_loss) #LR scheduler 

            val_loss_list.append(val_loss)
            val_acc_list.append(val_accuracy)
            if val_accuracy > best_val_acc + 0.001 :
                best_val_acc = val_accuracy
                best_epoch = epoch

                filename_best = f"best_{model_name}_{timestamp}.pth"
                #Save best model
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    'val_acc': val_accuracy
                 }, output_dir/filename_best)

            print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}')
            print(f'Epoch [{epoch+1}/{num_epochs}], Validation Accuracy: {val_accuracy:.4f}')
            print(f'Epoch [{epoch+1}/{num_epochs}], MCC: {mcc:.4f}')

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
        print(f"Best epoch is : {best_epoch+1}")

        

        filename = f"{model_name}_{timestamp}_{best_epoch+1}.pth"
        
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
         }, output_dir/filename)

        epochs = range(1, num_epochs +1)
        acc_title = f"Training-Validation_Accuracy_{model_name}"
        plt.title(acc_title)
        plt.plot(epochs,train_acc_list, label="Train Acc")
        plt.plot(epochs,val_acc_list, label="Val Acc")
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.xticks(range(0, num_epochs+1, 5))
        plt.ylim(0, 100)
        plt.legend()
        acc_fig = f"Training-Validation_Accuracy_{model_name}_{timestamp}.png"
        plt.savefig(os.path.join(output_dir, acc_fig),dpi=150)
        plt.close()

        loss_title = f"Training-Validation_Loss_{model_name}"
        plt.title(loss_title)
        plt.plot(epochs,train_loss_list, label="Train loss")
        plt.plot(epochs,val_loss_list, label="Val loss")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.xticks(range(0, num_epochs+1, 5))
        plt.legend()
        loss_fig = f"Training-Validation_Loss_{model_name}_{timestamp}.png"
        plt.savefig(os.path.join(output_dir, loss_fig),dpi=150)
        plt.close()
        
        return total_train_time,filename_best 

def get_model_cmplx(model, input_size=(3, 224, 224)):
    """
    Calculate computational metrics for trained model using torchvision summary
    
    Args:
        
        Model
        Input image size for the model
        
    Returns:
    
        Total trainable parameters in million
        GFlops 
    
    """
    model_summary = summary(
        model,
        input_size=(1, *input_size),
        verbose=0,
        col_names=["input_size", "output_size", "num_params", "mult_adds"]
    )
    
    trainable_params = model_summary.trainable_params / 1e6
    total_mac        = model_summary.total_mult_adds
    gflops           = total_mac / 1e9

    return trainable_params, gflops
    
def grad_camplus(file,model,model_name,output_dir,device):
        """
        Grad-CAM++ visualizations for the trained model.
        
        Selects 20 images from input data and produces a file with original image and 
        it's corresponding Grad-CAM++ activation map. File is saved in the output
        directory passed as parameter.
        
        Args:
            
           file       : File with validation or test data
           model      : Trained model to use for generating Grad-CAM++
           model_name : Model name
           output_dir : Directory to save the file
           device     : mps/gpu/cpu
        
        """

        CLASS_NAMES_STG1 = {
                             0 : "Non_Organic",
                             1 : "Organic"
        }

        data_df = pd.read_csv(file)
        rows_file = data_df.shape[0]
        data = data_df.iloc[:, :2].values.tolist()
		
        target_layers = [model.layer5]
		
        cam = GradCAMPlusPlus(
                               model=model,
                               target_layers=target_layers
                             )
							 
							 
        transform = v2.Compose([
            v2.Resize((224, 224)),
            #v2.CenterCrop(150),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )])
			
			
        cols = 4
        rows = math.ceil(len(data[0:20]) * 2 / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(16, 4*rows))
        axes = axes.flatten()
		
        val_file_name = os.path.basename(file) 
        if "val" in val_file_name.lower():
		
           cam_filename = f"cam_{model_name}_val_{timestamp}.png"
		
        else:
           cam_filename = f"cam_{model_name}_test_{timestamp}.png"
		
        for idx, (img_path, true_label) in enumerate(data[0:20]):
        
		        pil_img = Image.open(img_path).convert("RGBA").convert("RGB")
		        img_tensor = transform(pil_img).unsqueeze(0).to(device)
		        img_1 = transform(pil_img)
    
		        #Model prediction
		        with torch.no_grad():
		            output = model(img_tensor)
		            probs = torch.softmax(output, dim=1)
		            conf, pred = torch.max(probs, 1)
    
		        label = CLASS_NAMES_STG1[pred.item()]
		        conf_value = conf.item()
		        
		        targets = [ClassifierOutputTarget(pred.item())]

        
		        grayscale_cam = cam(input_tensor=img_tensor,targets=targets)[0]
        
		        rgb_img = np.array(pil_img.resize((224,224))).astype(np.float32) / 255.0
                #Activation map generation
		        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

		        color = "blue" if label.lower() == true_label.lower() else "red"
		        #Plot original and image with activation map 
		        orig_ax    = axes[idx * 2]
		        gradcam_ax = axes[idx * 2 + 1]
		        orig_ax.imshow(denormalize(img_1))
		        orig_ax.set_title(
		            f"Original: {true_label}",
		            color=color,
		            fontsize=14
		        )
		        orig_ax.axis("off")
		        
		        gradcam_ax.imshow(visualization)
		        gradcam_ax.set_title(
		            f"Pred: {label}\nConf:{conf_value:.2f}",
		            color=color,
		            fontsize=14
		        )
		        gradcam_ax.axis("off")
    
        #Hide empty cells if any
        for j in range(idx+1, len(axes)):
            axes[j].axis("off")
			
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, cam_filename),dpi=150)
        plt.close()

def eval_model(model,best_model_file,test_loader_s1,device,output_dir,class_names,train_time,val_file,test_file):
        """
        Main evaluation function that executes the full evaluation pipline
        
        1. Load best model checkpoint
        2. Calculate computational metrics
        3. Execute evaluation loop 
        4. Compute quantitative metrics and save 
        5. Plots : sample image validation, confusion matrix, classifiation report, Grad-CAM++
        6. Save all metrics file and plots in the output directort for analysis
        
        Args:
        
            model              : initialized model
            best_model_file    : file name of best model checkpoint
            test_loader_s1     : Test loader for stage1
            device             : device in which training is running (mps/gou etc)
            output_dir         : direcory to save files
            class_names        : Class names of input classes
            train_time         : Total training time retrieved from train function
            val_file           : File containing validation split data 
            test_file          : File containing test split data
        
        """    
        ckpt = torch.load(output_dir/best_model_file, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        
        # Validation
        model.eval()
        all_predictions = []
        all_labels = []
        start = time.time()
        
        trainable_params, gflops = get_model_cmplx(model, input_size=(3, 224, 224))
        #Main predictions
        with torch.no_grad():
            for images, labels, sources, type_images in test_loader_s1:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        end = time.time()
        num_images= len(test_loader_s1.dataset)
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

        images, labels, sources, type_images = next(iter(test_loader_s1))

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
        plt.savefig(os.path.join(output_dir, img_val_filename),dpi=150)
        plt.close()


        #Confusin Matrix
        cm = confusion_matrix(all_labels, all_predictions)
        fig, ax= plt.subplots()
        sns.heatmap(cm, annot=True, fmt='g', ax=ax,xticklabels=class_names,yticklabels=class_names);
        cm_title = f"Confusion Matrix Custom CNN"
        ax.set_xlabel('Predicted labels');
        ax.set_ylabel('True labels'); 
        ax.set_title(cm_title); 
        cm_filename = f"confusion_matrix_{model_name}_{timestamp}.png"
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, cm_filename),dpi=150,bbox_inches='tight')
        plt.close()

        #Classification report
        #Derive organic fp, macro/weighted F1 from CL report
        cr_filename = f"classification_report_{model_name}_{timestamp}.csv"
        tn, fp, fn, tp= confusion_matrix(all_labels, all_predictions).ravel()
        organic_false_positive_rate = fp / (fp + tn)
        report = classification_report(all_labels, all_predictions,output_dict=True,
                                       target_names=class_names)
        report_df = pd.DataFrame(report).transpose()
        report_df_classes = report_df.iloc[:len(class_names)]
        report_df_agg = report_df.iloc[len(class_names):]
        macro_f1 = report_df_agg.loc["macro avg", "f1-score"]
        weighted_f1 = report_df_agg.loc["weighted avg", "f1-score"]
        report_df_classes.to_csv(output_dir/cr_filename)

        #Derive class wise precision/recall/F1
        non_org_precision = round( report["Non_Organic"]["precision"] * 100, 2)
        non_org_recall = round( report["Non_Organic"]["recall"] * 100 , 2)
        non_org_f1 = round(report["Non_Organic"]["f1-score"] * 100,2)
        org_precision = round(report["Organic"]["precision"] * 100,2)
        org_recall = round(report["Organic"]["recall"] * 100,2)
        org_f1 = round(report["Organic"]["f1-score"] * 100,2)

            
        total_params = sum(p.numel() for p in model.parameters())
        
        #Save evaluation metrics
        eval_metrics_df = pd.DataFrame({
        "params(M)" : [round(total_params/1000000,2)], 
        "trainingtime(min)" : [round(train_time/60,2)],
        "accuracy": [round(accuracy,2)],
        "precision": [round(precision * 100,2)],
        "recall": [round(recall * 100,2)],
        "f1_score": [round(f1 * 100,2)],
        "balanced_accuracy" : [round(balanced_acccuracy,2)],
        "MCC": [round(mcc,2)],
        "Inference time per image" : [round(avg_time*1000,2)],
        "Throughput" : [round(throughput,2)],
        "Trainable_params(M)" : [round(trainable_params,2)],
        "GFLOPs" : [round(gflops,2)],
        "organic_precision": [org_precision],
        "organic_recall": [org_recall],
        "organic_F1" : [org_f1],
        "Norganic_precision": [non_org_precision],
        "Norganic_recall": [non_org_recall],
        "Norganic_F1" : [non_org_f1],
        "Organic_FP" : [round(organic_false_positive_rate*100,2)],
        "Macro_F1" : [round(macro_f1*100,2)]
        })

        eval_metrics_file = f"eval_metrics_{model_name}_{timestamp}.csv"
        eval_metrics_df.to_csv(output_dir/eval_metrics_file, index=False)
		
	    #Grad-CAM eval of 20 images from val and test dataset
		
        grad_camplus(val_file,model,model_name,output_dir,device)
        grad_camplus(test_file,model,model_name,output_dir,device)
		
def main():
        """
        Initial main function which executes the E2E pipeline 
        
        1. Load metadata file for dataset and dataloader setup
        2. Stratified split
        3. Define phasewise train and validation torchvision transform
        4. Dataset creation using respective transform (train/val/test)
        5. Weightedrandomsampler for class balancing
        6. Dataloader creation for train/val/test
        7. Model initialization
        8. Train function call
        9. Eval function call
        
        """    
        os.getcwd()
        #os.chdir("..")
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

        #Input data location and folders config
        os.chdir('/home/968001/waste_classification/')
        BASE_DIR = Path.cwd()
        META_DIR = BASE_DIR / "data" / "metadata"
        file1=META_DIR/"consolidated_metatdata-stg1.csv"
        output_dir = BASE_DIR / "output" / "Stage1" / "cnn" /  "V8"
        val_file = output_dir/"val_data.csv"
        test_file = output_dir/"test_data.csv"

        metadata_df = pd.read_csv(file1)

        #Stratified split
        train_df, temp_df = train_test_split(
            metadata_df,
            test_size=0.30,          # 70 train / 30 temp
            stratify=metadata_df["stage1_label"],
            random_state=42
        )

        val_df, test_df = train_test_split(
            temp_df,
            test_size=0.50,          # 15 / 15
            stratify=temp_df["stage1_label"],
            random_state=42
        )

        val_df.to_csv(val_file, index=False)
        test_df.to_csv(test_file, index=False)
        
        #Relaying back the split to the metadata dataframe
        metadata_df.loc[metadata_df["filepath"].isin(train_df["filepath"]), "split"] = "train"
        metadata_df.loc[metadata_df["filepath"].isin(test_df["filepath"]), "split"] = "test"
        metadata_df.loc[metadata_df["filepath"].isin(val_df["filepath"]), "split"] = "val"

        val_df = metadata_df.loc[metadata_df['split'] == "val"].copy()
        #val_df["dup_flag"] = 0

        test_df = metadata_df.loc[metadata_df['split'] == "test"].copy()
        #test_df["dup_flag"] = 0

        organic_df = metadata_df.loc[ (metadata_df['stage1_label'] == "Organic") & (metadata_df['split'] == "train") ].copy()
        #organic_df["dup_flag"] = 0
        organic_count = organic_df.shape[0]

        # Duplicate
        #organic_dup = organic_df.copy()
        #organic_dup["dup_flag"] = 1

        nonorganic_df = metadata_df.loc[(metadata_df['stage1_label'] == "Non_Organic") & (metadata_df['split'] == "train")].copy()
        #nonorganic_df["dup_flag"] = 0
        non_organic_count = nonorganic_df.shape[0]

        updated_metadata_df = pd.concat(
            [organic_df,nonorganic_df,val_df, test_df],
            ignore_index=True
        ).sample(frac=1)

        updated_metadata_df['stage1_label'].value_counts()


        #Define transforms for train, val and test
        train_transforms = {
            "standard": v2.Compose([
            v2.RandomResizedCrop((192,192),scale=(0.5, 1.0)),
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            v2.RandomRotation(30),
            v2.RandomPerspective(distortion_scale=0.2, p=0.3),
            v2.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.2,hue=0.1
            ),
            v2.RandomGrayscale(p=0.1),
            v2.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            v2.RandomErasing(p=0.2, scale=(0.02, 0.2))
            ])
        }

        train_transforms_2 = {
            "standard": v2.Compose([
            v2.RandomResizedCrop((224,224),scale=(0.5, 1.0)),
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            v2.RandomRotation(30),
            v2.RandomPerspective(distortion_scale=0.2, p=0.3),
            v2.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.2,hue=0.1
            ),
            v2.RandomGrayscale(p=0.1),
            v2.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            v2.RandomErasing(p=0.2, scale=(0.02, 0.2))
            ])
        }


        val_transform = {
            
            "default": v2.Compose([
            v2.Resize((192, 192)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
           ])
        }

        val_transform_2 = {
            
            "default": v2.Compose([
            v2.Resize((224, 224)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
           ])
        }
		
        #Creating datasets - Stage 1 
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

        test_ds_stage1 = CustomWasteData(
            updated_metadata_df,
            split="test",
            label_col="stage1_label",
            transforms_dict=val_transform_2,
            train=False
        )
        
        #class_labels = [train_ds_stage1[i][1] for i in range(len(train_ds_stage1))]

        #WeightedRandomSampler 
        #Retrieving integer class label for every sample
        class_labels =  train_ds_stage1.targets
        class_labels = np.array(class_labels)
        class_counts = [non_organic_count, organic_count]
        #Class weights as inverse frequency
        class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
        #Assign per-sample weight based on class weights
        sample_weights = class_weights[class_labels]  
        #WeightedRandomSampler with replacement true to balance minority and majority class samples
        w_sampler = WeightedRandomSampler(sample_weights, len(sample_weights),replacement=True)
        
        #Dataloader setup for Train/Val/Test
        train_loader_s1 = DataLoader(
            train_ds_stage1,
            batch_size=64,
            #shuffle=True,
            sampler=w_sampler, 
            num_workers=4,
            pin_memory=True,
            persistent_workers=False,
            drop_last=True
        )

        val_loader_s1 = DataLoader(
            val_ds_stage1,
            batch_size=64,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=False
        )

        test_loader_s1 = DataLoader(
            test_ds_stage1,
            batch_size=64,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=False
        )

        class_names = val_ds_stage1.classes
        
        print('--------------------------------')
        print("CNN Model V8 eval and training"  )
        print('--------------------------------')
        CNN_stage1_res_model = CNN_stage1_res(2).to(device)
        #Train model on train plus epoch wise evaluation on val set
        train_time, best_model_file = train_model(CNN_stage1_res_model,train_loader_s1,val_loader_s1,device,output_dir,train_transforms,val_transform,
		                                          train_transforms_2,val_transform_2,train_ds_stage1,val_ds_stage1)
        #Final evaluation on test set
        eval_model(CNN_stage1_res_model,best_model_file,test_loader_s1,device,output_dir,class_names,train_time,val_file,test_file)
        


if __name__ == "__main__":
    main()