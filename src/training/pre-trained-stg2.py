#!/usr/bin/env python
# coding: utf-8

#### Pretrained DenseNet121,ResNet34,MobileNetV2 Stage 2 - Training and Evaluation script

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
from torch.utils.data import default_collate
from collections import Counter
from torch.utils.data import WeightedRandomSampler
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
import math
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchinfo import summary
from torchmetrics import Accuracy

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Custom wastedataset for Dataloader

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
    def __init__(self, csv_df, split,label_col, transform_dict,train=True):
        self.df = csv_df
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        
        self.transforms_dict = transform_dict
        self.label_col = label_col
        self.train = train
        
        self.classes = sorted(self.df[label_col].unique())
        self.class_to_idx = {
            cls: i for i, cls in enumerate(self.classes)
        }
        self.targets = [self.class_to_idx[l] for l in self.df[label_col].tolist()]

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        
        try:
            row = self.df.iloc[idx]
    
            img = Image.open(row["filepath"]).convert("RGBA").convert("RGB")    
            label = self.class_to_idx[row[self.label_col]]
            #label = self.targets[idx]
            source = row["source"]     
            type_image = row["type"]    
            # Different augmentations for studio predominant classes
            if self.train:
                if row['clean_aug'] == 1:
                    img = self.transforms_dict["clean"] (img)
                else:
                    img = self.transforms_dict["standard"] (img)
            else:
                img = self.transforms_dict["default"] (img)
                
    
            return img, label, source, type_image
        except Exception as e:
            print(f"Error at idx {idx}: {row['filepath']} — {e}")
            raise

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


class CNN_stage2_r34(torch.nn.Module):
    """
    ResNet34 with a custom two-layer fully connected head for waste classification using 
    progressive unfreezing and adaptive learning rates.
    
    Classifier head customized with dropout before each dense layer. Batchnormalization
    and activation applied in between dense layers.
    
    Initial retraining only on classifier. Progressive unfreezing from layer 4(phase 2) upto 
    layer 2(phase 4)
    
    Args
    
        num_classes : number of classes : 2 for stage1 (int)
        input_size  : Input image size. Default to 224x224
        channels    : input channels
        
    """
    def __init__(self, num_classes, input_size=(256, 256), channels=3):
        super(CNN_stage2_r34, self).__init__()

        self.input_size = input_size
        self.channels = channels

        self.network = resnet34(weights=ResNet34_Weights.DEFAULT)

        num_features = self.network.fc.in_features
        self.network.fc = torch.nn.Sequential(
                                torch.nn.Dropout(p=0.3),
                                torch.nn.Linear(num_features, 1024),
                                torch.nn.BatchNorm1d(1024),
                                torch.nn.ReLU(inplace=True),
                                torch.nn.Dropout(0.4),
                                torch.nn.Linear(1024, num_classes)
                                )

    def forward(self, x):
        return self.network(x)

    def freeze(self):

        # Freeze all layers
        for param in self.network.parameters():
            param.requires_grad = False

        # Unfreeze final classifier layer
        for param in self.network.fc.parameters():
            param.requires_grad = True

    def unfreeze(self):

        for param in self.network.parameters():
            param.requires_grad = True



class CNN_stage2_mnv2(torch.nn.Module):
    """
    MobilenetV2 with a custom two-layer fully connected head for waste classification using 
    progressive unfreezing and adaptive learning rates.
    
    Classifier head customized with dropout before each dense layer. Batchnormalization
    and activation applied in between dense layers.
    
    Initial retraining only on classifier. Progressive unfreezing from IR block 15-17 (phase 1)
    all the way up to IR block 1-17 (phase 4)
    
    Args
    
        num_classes : number of classes : 2 for stage1 (int)
        input_size  : Input image size. Default to 224x224
        channels    : input channels
        
    """
    def __init__(self, num_classes, input_size=(256, 256), channels=3):
        super(CNN_stage2_mnv2, self).__init__()

        self.input_size = input_size
        self.channels = channels

        self.network = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)

        num_features = self.network.classifier[1].in_features
        self.network.classifier = torch.nn.Sequential(
                                torch.nn.Linear(num_features, 1024),
                                torch.nn.BatchNorm1d(1024),
                                torch.nn.ReLU(inplace=True),
                                torch.nn.Dropout(0.4),
                                torch.nn.Linear(1024, num_classes)
                                )

    def forward(self, x):
        return self.network(x)

    def freeze(self):

        # Unfreeze all layers
        for param in self.network.parameters():
            param.requires_grad = False

        # Unfreeze final classifier layer
        for param in self.network.classifier.parameters():
            param.requires_grad = True

    def unfreeze(self):

        for param in self.network.parameters():
            param.requires_grad = True


class CNN_stage2_dn121(torch.nn.Module):
    """
    DenseNet121 with a custom two-layer fully connected head for waste classification using 
    progressive unfreezing and adaptive learning rates.
    
    Classifier head customized with dropout before each dense layer. Batchnormalization
    and activation applied in between dense layers.
    
    Initial retraining only on classifier.Progressive unfreezing from Denseblock4(phase 2) upto 
    Denseblock2(phase 4)
    
    Args
    
        num_classes : number of classes : 2 for stage1 (int)
        input_size  : Input image size. Default to 224x224
        channels    : input channels
        
    """
    def __init__(self, num_classes, input_size=(224, 224), channels=3):
        super(CNN_stage2_dn121, self).__init__()

        self.input_size = input_size
        self.channels = channels

        self.network = densenet121(weights=DenseNet121_Weights.DEFAULT)

        num_features = self.network.classifier.in_features
        self.network.classifier = torch.nn.Sequential(
                                torch.nn.Dropout(p=0.3),
                                torch.nn.Linear(num_features, 1024),
                                torch.nn.BatchNorm1d(1024),
                                torch.nn.ReLU(inplace=True),
                                torch.nn.Dropout(0.4),
                                torch.nn.Linear(1024, num_classes)
                                )

    def forward(self, x):
        return self.network(x)

    def freeze(self):

        # Freeze all layers
        for param in self.network.parameters():
            param.requires_grad = False

        # Unfreeze final classifier layer
        for param in self.network.classifier.parameters():
            param.requires_grad = True

    def unfreeze(self):

        for param in self.network.parameters():
            param.requires_grad = True
            

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

NUM_CLASSES=13

cutmix = v2.CutMix(num_classes=NUM_CLASSES)
mixup = v2.MixUp(num_classes=NUM_CLASSES)
#cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])

def get_prob(epoch):
    """
        CutMix/Mixup probability configuration for each phase
        Returns probability based on epoch as input
        
    """
    if epoch < 20:
        mix_prob = 0.2
    elif epoch < 45:
        mix_prob = 0.5
    elif epoch < 75:
        mix_prob = 0.4
    else:
        mix_prob = 0.2
    
    return mix_prob
    
def get_optimizer_and_scheduler(model, model_name, phase):
    """
    Phase-wise optimizer and scheduler setup
    Phase 1 :  0 to 19
    Phase 2 : 20 to 44
    Phase 3 : 45 to 74
    Phase 4 : 75 to 100
    
    Args:
        
        model      : Model being trained
        model_name : Model name
        Phase      : 1-4
        
    Returns:
    
        Initialized optimizer, scheduler
    
    """
    #LR selected based on tuning experiments for best performance
    #Higher LR for classifier with progressive decrease as we move more towards
    #generic layers
    if 'dn121' in model_name:
        if phase == 1:
            optimizer = torch.optim.AdamW([
            {'params': list(model.network.classifier.parameters()),            'lr': 5e-4, 'weight_decay': 1e-4}
            ])
        elif phase == 2:
            params_list = list(model.network.features.denseblock4.parameters()) + \
                          list(model.network.features.norm5.parameters())
            optimizer = torch.optim.AdamW([
            {'params': list(model.network.features.denseblock4.parameters()) +
                       list(model.network.features.norm5.parameters()),        'lr': 1e-4, 'weight_decay': 1e-4},
            {'params': list(model.network.classifier.parameters()),            'lr': 5e-4, 'weight_decay': 1e-4}
            ])
        elif phase == 3:
            params_list = list(model.network.features.denseblock3.parameters()) + \
                          list(model.network.features.transition3.parameters())
            optimizer = torch.optim.AdamW([
            {'params': list(model.network.features.denseblock3.parameters()) +
                       list(model.network.features.transition3.parameters()),  'lr': 1e-4, 'weight_decay': 1e-4}, #changed from 5e-5
            {'params': list(model.network.features.denseblock4.parameters()) +
                       list(model.network.features.norm5.parameters()),        'lr': 1e-4, 'weight_decay': 1e-4},
            {'params': list(model.network.classifier.parameters()),            'lr': 5e-4, 'weight_decay': 1e-4}
            ])
        elif phase == 4:
            params_list = list(model.network.features.denseblock2.parameters()) + \
                          list(model.network.features.transition2.parameters())
            optimizer = torch.optim.AdamW([
            {'params': list(model.network.features.denseblock2.parameters()) +
                       list(model.network.features.transition2.parameters()),  'lr': 1e-5, 'weight_decay': 1e-5},
            {'params': list(model.network.features.denseblock3.parameters()) +
                       list(model.network.features.transition3.parameters()),  'lr': 1e-4, 'weight_decay': 1e-4},#changed from 5e-5
            {'params': list(model.network.features.denseblock4.parameters()) +
                       list(model.network.features.norm5.parameters()),        'lr': 1e-4, 'weight_decay': 1e-4},
            {'params': list(model.network.classifier.parameters()),            'lr': 5e-4, 'weight_decay': 1e-4}
            ])
    elif 'mnv2' in model_name:
        if phase == 1:
            optimizer = torch.optim.AdamW([
                {'params': list(model.network.classifier.parameters()),     'lr': 5e-4, 'weight_decay': 1e-4}
            ])
        elif phase == 2:
            params_list = list(model.network.features[15:].parameters())
            optimizer = torch.optim.AdamW([
                {'params': list(model.network.features[15:].parameters()),   'lr': 1e-4, 'weight_decay': 1e-4},
                {'params': list(model.network.classifier.parameters()),       'lr': 5e-4, 'weight_decay': 1e-4}
            ])
        elif phase == 3:
            params_list = list(model.network.features[8:15].parameters())
            optimizer = torch.optim.AdamW([
                {'params': list(model.network.features[8:15].parameters()),  'lr': 5e-5, 'weight_decay': 1e-4},
                {'params': list(model.network.features[15:].parameters()),   'lr': 1e-4, 'weight_decay': 1e-4},
                {'params': list(model.network.classifier.parameters()),       'lr': 5e-4, 'weight_decay': 1e-4}
            ])
        elif phase == 4:
            params_list = list(model.network.features[1:8].parameters())	
            optimizer = torch.optim.AdamW([
                {'params': list(model.network.features[1:8].parameters()),   'lr': 1e-5, 'weight_decay': 1e-5},
                {'params': list(model.network.features[8:15].parameters()),  'lr': 5e-5, 'weight_decay': 1e-4},
                {'params': list(model.network.features[15:].parameters()),   'lr': 1e-4, 'weight_decay': 1e-4},
                {'params': list(model.network.classifier.parameters()),       'lr': 5e-4, 'weight_decay': 1e-4}
            ])

    elif 'r34' in model_name:
        if phase == 1:
            optimizer = torch.optim.AdamW([
                {'params': list(model.network.fc.parameters()),     'lr': 5e-4, 'weight_decay': 1e-4}
            ])
        elif phase == 2:
            params_list = list(model.network.layer4.parameters())
            optimizer = torch.optim.AdamW([
                {'params': list(model.network.layer4.parameters()), 'lr': 1e-4, 'weight_decay': 1e-4},
                {'params': list(model.network.fc.parameters()),     'lr': 5e-4, 'weight_decay': 1e-4}
            ])
        elif phase == 3:
            params_list = list(model.network.layer3.parameters())
            optimizer = torch.optim.AdamW([
                {'params': list(model.network.layer3.parameters()), 'lr': 5e-5, 'weight_decay': 1e-4},
                {'params': list(model.network.layer4.parameters()), 'lr': 1e-4, 'weight_decay': 1e-4},
                {'params': list(model.network.fc.parameters()),     'lr': 5e-4, 'weight_decay': 1e-4}
            ])
        elif phase == 4:
            params_list = list(model.network.layer2.parameters())
            optimizer = torch.optim.AdamW([
                {'params': list(model.network.layer2.parameters()), 'lr': 1e-5, 'weight_decay': 1e-5},
                {'params': list(model.network.layer3.parameters()), 'lr': 5e-5, 'weight_decay': 1e-4},
                {'params': list(model.network.layer4.parameters()), 'lr': 1e-4, 'weight_decay': 1e-4},
                {'params': list(model.network.fc.parameters()),     'lr': 5e-4, 'weight_decay': 1e-4}
            ])

    #Different patience configuration baseed on progressive training phase
    #To allow LR adjustments based on training stage
    if phase in [1, 2]:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5,
            patience=5, min_lr=1e-7
        )
    elif phase == 3:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.3,
            patience=4, min_lr=1e-7
        )
    else:  
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.3,
            patience=3, min_lr=1e-7
        )
    if phase in [2,3,4]:
       for param in params_list:
           param.requires_grad = True
    #Validate trainable parameters phase wise         
    for name, p in model.named_parameters():
       if p.requires_grad:
          print("Trainable:", name)

    return optimizer, scheduler

def train_model(model,train_loader_s2,val_loader_s2,device,output_dir, train_ds_stage2, val_ds_stage2,
                train_transform_1,train_transform_2,train_transform_3,train_transform_4,val_transform_1,
                val_transform_2,val_transform_3,class_weights_norm):
        """
        Main training function that executes the full training pipline
        
        1. Training configuration and parameter setup including adaptive learning rate setup 
        2. Progressive resizing and augmentation (cutmix)
        3. Handling progressive training phases switch
        4. Training and validation loops for number of epochs
        5. Training and validation metric computation
        6. Early stopping and best model checkpointing
        7. Training/validation accuracy loss plots
        
        Args:
        
            model                     : initialized model
            train_loader_s2           : Train loader for stage2
            val_loader_s2             : Val loader for stage2
            device                    : device in which training is running (mps/gou etc)
            output_dir                : direcory to save files
            train_ds_stage2           : Train dataset stage2
            val_ds_stage2             : val dataset stage2
            train_transforms_1 to 4   : Torchvision transform for training data for each phase
            val_transform_1 to 3      : Torchvision transform for val data for each phase
            class_weights_norm        : Normalized class weights for use in loss function
            
    
        Returns:
        
            total_train_time : Total training time in seconds
            best_model_file  : Filename of the best model checkpoint
        
        """

        num_epochs = 100
        model_name = model.__class__.__name__
        #Label smoothing to regularize model prediction overconfidence
        #Class weight to regularize minority classes prediction
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1,weight=class_weights_norm.to(device))

        model.freeze()
        #Initial early stopping patience for 12 epochs and then reduced in later stages
        early_stopping = EarlyStopping(patience=12, delta=0.001)


        # Training loop
        epoch_times = []
        train_acc_list = []
        train_loss_list = []
        val_acc_list = []
        val_loss_list= []
        best_val_acc = 0
        best_epoch = 0
        
        print(train_ds_stage2.transforms_dict)
        
        
        for epoch in range(num_epochs):
            start = time.time()
            model.train()
            running_loss = 0.0
            train_loss = 0.0
            train_accuracy = 0.0
            all_predictions = []
            all_labels = []
            
            #Retrieve probability for cutmix/mixup
            mix_prob = get_prob(epoch)
            #3:1 ratio for cutmix vs mixup
            mix_transform = v2.RandomApply(
                [v2.RandomChoice([cutmix,cutmix,cutmix,mixup])],
                p=mix_prob
            )
           
            #Phase 1    
            if epoch == 0:
                # Retrieve optimizer/scheduler and switch transform
                optimizer,scheduler = get_optimizer_and_scheduler(model,model_name,1)

                train_ds_stage2.transforms_dict = train_transform_1
                val_ds_stage2.transforms_dict = val_transform_1
                print("Reset transform")
            #Phase 2 
            elif epoch == 20:
                # Retrieve optimizer/scheduler and switch transform
                optimizer,scheduler = get_optimizer_and_scheduler(model,model_name,2)

                train_ds_stage2.transforms_dict = train_transform_2
                val_ds_stage2.transforms_dict = val_transform_2
                print("Switched to phase 2 transform")
                
                print(train_ds_stage2.transforms_dict)
                
                for images, labels, sources, type_images in train_loader_s2:
                    print("Train batch shape:", images.shape)
                    break
                        
                early_stopping = EarlyStopping(patience=8, delta=0.001)
            #Phase 3    
            if epoch == 45:
                # Retrieve optimizer/scheduler and switch transform
                optimizer,scheduler = get_optimizer_and_scheduler(model,model_name,3)

                train_ds_stage2.transforms_dict = train_transform_3
                val_ds_stage2.transforms_dict = val_transform_3
                print("Switched to phase 3 transform")
                
                print(train_ds_stage2.transforms_dict)

                for images, labels, sources, type_images in train_loader_s2:
                    print("Train batch shape:", images.shape)
                    break
                
                early_stopping = EarlyStopping(patience=6, delta=0.001)
            #Phase 4              
            if epoch == 75:
                # Retrieve optimizer/scheduler and switch transform
                optimizer,scheduler = get_optimizer_and_scheduler(model,model_name,4)


                train_ds_stage2.transforms_dict = train_transform_4
                val_ds_stage2.transforms_dict = val_transform_3
                print("Switched to phase 4 transform")
                
                print(train_ds_stage2.transforms_dict)
                
                for images, labels, sources, type_images in train_loader_s2:
                    print("Train batch shape:", images.shape)
                    break
           
                        
                early_stopping = EarlyStopping(patience=5, delta=0.001)
            
            for images, labels, sources, type_images in train_loader_s2:
                #Cutmix/mixup applied batchwise  
                images, labels = mix_transform(images, labels)
                
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
            
            model.eval()
            all_predictions = []
            all_labels = []
            running_loss = 0.0
            val_loss = 0.0
            val_accuracy = 0.0
            for images, labels, sources, type_images in val_loader_s2:
                print("Val batch shape:", images.shape)
                break
        
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
            mcc = matthews_corrcoef(all_labels, all_predictions)

            val_loss_list.append(val_loss)
            val_acc_list.append(val_accuracy)

            if val_accuracy > best_val_acc + 0.001 :
                best_val_acc = val_accuracy
                best_epoch = epoch

                filename_best = f"best_{model_name}_{timestamp}.pth"
                
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    'val_acc': val_accuracy
                 }, output_dir/filename_best)

            print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}')
            print(f'Epoch [{epoch+1}/{num_epochs}], Validation Accuracy: {val_accuracy:.4f}')
            
            scheduler.step(val_loss)

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
        
        Target layer configured based on model
        
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
        CLASS_NAMES_STG2 = {
            0 : "Battery",
            1 : "Brown_Glass",
            2 : "Cardboard",
            3 : "E_Waste",
            4 : "Green_Glass",
            5 : "Medical_Waste",
            6 : "Metal",
            7 : "Misc_Trash",
            8 : "Paper",
            9 : "Plastic",
            10 : "Shoes",
            11 : "Textile",
            12 : "White_Glass"
        }


        data_df = pd.read_csv(output_dir/file)
        rows_file = data_df.shape[0]
        data = data_df.iloc[:, [0, 2]].values.tolist()

        if "r34" in model_name: 

            target_layers = [model.network.layer4]

        elif "mnv2" in model_name:

            target_layers = [model.network.features[-1]]

        elif "dn121" in model_name:

            target_layers = [model.network.features.denseblock4]

        cam = GradCAMPlusPlus(
                               model=model,
                               target_layers=target_layers
                             )

        transform = v2.Compose([
            v2.Resize((320, 320)),
            #v2.CenterCrop(150),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )])
			
			
        cols = 4
        rows = math.ceil(len(data[0:20] * 2) / cols)

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
    
                label = CLASS_NAMES_STG2[pred.item()]
                conf_value = conf.item()
                targets = [ClassifierOutputTarget(pred.item())]
                # if label.lower() != true_label.lower():
        

                grayscale_cam = cam(input_tensor=img_tensor,targets=targets)[0]
        
                rgb_img = np.array(pil_img.resize((320,320))).astype(np.float32) / 255.0
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


def eval_model(model,best_model_file,test_loader_s2,device,output_dir,class_names, train_time,val_file,test_file):
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
        top_2_accuracy = Accuracy(task="multiclass", num_classes=13, top_k=2).to(device)
        
        # Validation
        model.eval()
        all_predictions = []
        all_labels = []
        start = time.time()
	
        trainable_params, gflops = get_model_cmplx(model, input_size=(3, 320, 320))
        
        for images, labels, sources, type_images in test_loader_s2:
            print("Final Val batch shape:", images.shape)
            break
        
        with torch.no_grad():
            for images, labels, sources, type_images in test_loader_s2:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                top_2_accuracy.update(probs, labels) 
                _, predicted = torch.max(outputs.data, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        end = time.time()
        num_images= len(test_loader_s2.dataset)
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
        if "r34" in model_name: 

            model_name_txt = 'ResNet34'

        elif "mnv2" in model_name:
            
            model_name_txt = 'MobilenetV2'

        elif "dn121" in model_name:

            model_name_txt = 'DenseNet121'

        images, labels, sources, type_images = next(iter(test_loader_s2))

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
        #For readability class names are abbreviated
        abbrev_class = {
        'Battery' : 'Battery',
        'Brown_Glass' : 'BGlass',
        'Cardboard' : 'Cboard',
        'E_waste' : 'Ewaste',
        'Green_Glass': 'GGlass',
        'Medical_waste': 'Medic',
        'Metal': 'Metal',
        'Misc_Trash' : 'Trash',
        'Paper': 'Paper',
        'Plastic': 'Plastic',
        'Shoes': 'Shoes',
        'Textile': 'Textile',
        'White_Glass': 'WGlass'
        }
        
        clss_abbr_names = [abbrev_class[c] for c in class_names]	
        cm = confusion_matrix(all_labels, all_predictions,)
        fig, ax= plt.subplots()
        sns.heatmap(cm, annot=True, fmt='g', ax=ax,xticklabels=clss_abbr_names,yticklabels=clss_abbr_names);
        cm_title = f"Confusion Matrix {model_name_txt}"
        ax.set_xlabel('Predicted labels');
        ax.set_ylabel('True labels'); 
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        ax.set_title(cm_title); 
        cm_filename = f"confusion_matrix_{model_name}_{timestamp}.png"
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, cm_filename),dpi=150,bbox_inches='tight')
        plt.close()
        
        #Classification report
        cr_filename = f"classification_report_{model_name}_{timestamp}.csv"
        
        report = classification_report(all_labels, all_predictions,output_dict=True,
                                       target_names=class_names)

        # Extract per-class F1
        f1_scores = [report[cls]["f1-score"] for cls in reversed(class_names)]
        fig, ax= plt.subplots()
        
        bars = ax.barh(list(reversed(clss_abbr_names)), f1_scores, color=[
                      "green" if f >= 0.90 else "orange" if f >= 0.75 else "red" 
                       for f in f1_scores
                      ])
                      
        for bar, f1_s in zip(bars, f1_scores):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f"{f1_s:.2f}", va="center", fontsize=10)
            
        ax.set_xlim(0, 1.1)
        ax.set_xlabel("F1-Score")
        f1_title = f"Per-Class F1 Scores {model_name_txt}"
        ax.set_title(f1_title)
        #Threshold for green line at 0.90
        ax.axvline(x=0.90, color="green", linestyle="--", alpha=0.5, label="0.90 threshold")
        fig.tight_layout()
        
        pc_f1_filename = f"per_class_F1_{model_name}_{timestamp}.png"
        fig.savefig(os.path.join(output_dir, pc_f1_filename), dpi=150,bbox_inches="tight")
        plt.close()
        
        #Eval metrics

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
        "top2_accuracy" : [round(top_2_accuracy.compute().item() * 100,2)],
        "precision": [round(precision * 100,2)],
        "recall": [round(recall * 100,2)],
        "f1_score": [round(f1 * 100,2)],
        "balanced_accuracy" : [round(balanced_acccuracy,2)],
        "MCC": [round(mcc,2)],
        "Inference time per image" : [round(avg_time*1000,2)],
        "Throughput" : [round(throughput,2)],
        "Trainable_params(M)" : [round(trainable_params,2)],
        "GFLOPs" : [round(gflops,2)],
        "Macro F1" : [round(macro_f1,2)]
        })
        #Save Eval metrics
        eval_metrics_file = f"eval_metrics_{model_name}_{timestamp}.csv"
        eval_metrics_df.to_csv(os.path.join(output_dir, eval_metrics_file), index=False)
        
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
        5. Weightedrandomsampler for class balancing, normalized class weights
        6. Dataloader creation for train/val/test
        7. Model initialization
        8. Train function call
        9. Eval function call (one by one for each pretrained model)
        
        """
        os.getcwd()
        #os.chdir("..")

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
        #Input image files for training were kept in viper user home location 
        #with directory structure /home/968001/waste_classification/dara
        
        os.chdir('/home/968001/waste_classification/')
        
        BASE_DIR = Path.cwd()
        META_DIR = BASE_DIR / "data" / "metadata"
        file1=META_DIR/"consolidated_metatdata-stg2.csv"
        output_dir = BASE_DIR / "output" / "Stage2" / "pretrained" / "V9"
        
        val_file =  f"val_data_{timestamp}.csv"
        test_file = f"test_data_{timestamp}.csv"

        metadata_df = pd.read_csv(file1)
        #Filter only Stage2 rows i.e with stage2 labels present
        stg2_meta_df = metadata_df.loc[~metadata_df['stage2_label'].isin(['Non_Organic','Organic'])].copy()
        
        #Stratified split
        train_df, temp_df = train_test_split(
            stg2_meta_df,
            test_size=0.30,          # 70 train / 30 temp
            stratify=stg2_meta_df["stage2_label"],
            random_state=42
        )

        val_df, test_df = train_test_split(
            temp_df,
            test_size=0.50,          # 15 / 15pand
            stratify=temp_df["stage2_label"],
            random_state=42
        )
        #Relaying back the split to the metadata dataframe
        stg2_meta_df.loc[stg2_meta_df["filepath"].isin(train_df["filepath"]), "split"] = "train"
        stg2_meta_df.loc[stg2_meta_df["filepath"].isin(test_df["filepath"]), "split"] = "test"
        stg2_meta_df.loc[stg2_meta_df["filepath"].isin(val_df["filepath"]), "split"] = "val"
        
        #Flag for distinguishing stduio only classes for additional augmentation to be applied
        stg2_meta_df["clean_aug"] = 0
        stg2_meta_df.loc[(stg2_meta_df["split"] == "train") & (stg2_meta_df["stage2_label"].isin(['Battery','E_waste','Medical_waste'])),"clean_aug"] == 1

        val_df.to_csv(output_dir/val_file, index=False)
        test_df.to_csv(output_dir/test_file, index=False)
        
        #Define transforms for train, val and test
        #Phase wise transform to handle Progressive resizing and augmentation
        train_transform_1 = { 
            "standard" : v2.Compose([
            #v2.Resize((256, 256)),
            #v2.PILToTensor(),
            v2.RandomResizedCrop(224, scale=(0.5,1.0),antialias=True),
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(p=0.3),
            v2.RandomRotation(15),
            v2.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2
            ),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            v2.RandomErasing(p=0.4)
            ]),
            "clean" : v2.Compose([
            v2.RandomResizedCrop(224, scale=(0.5,1.0), antialias=True),
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(p=0.3),
            v2.RandomRotation(20),
            v2.ColorJitter(0.3, 0.3, 0.3),
            # Extra studio augmentations before normalize
            v2.GaussianBlur(5, (0.1, 2)),
            #v2.RandomPerspective(0.3, 0.5),
            v2.RandomAutocontrast(p=0.2),
            v2.RandomGrayscale(p=0.15),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            v2.RandomErasing(p=0.5)
            ])
            
        }

        train_transform_2 = {
            "standard" : v2.Compose([
            #v2.Resize((256, 256)),
            #v2.PILToTensor(),
            v2.RandomResizedCrop(256, scale=(0.5,1.0),antialias=True),
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(p=0.3),
            v2.RandomRotation(12),
            v2.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.15
            ),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            v2.RandomErasing(p=0.3)
            ]),
            "clean" : v2.Compose([
            v2.RandomResizedCrop(256, scale=(0.5,1.0), antialias=True),
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(p=0.3),
            v2.RandomRotation(17),
            v2.ColorJitter(0.25, 0.25, 0.25),
            # Extra studio augmentations before normalize
            v2.GaussianBlur(5, (0.1, 1.5)),
            #v2.RandomPerspective(0.2, 0.4),
            v2.RandomAutocontrast(p=0.15),
            v2.RandomGrayscale(p=0.1),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            v2.RandomErasing(p=0.4)
            ])
            
        }
            
            
        train_transform_3 = {
            "standard" : v2.Compose([
            #v2.Resize((256, 256)),
            #v2.PILToTensor(),
            v2.RandomResizedCrop(320, scale=(0.6,1.0),antialias=True),
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(p=0.3),
            v2.RandomRotation(10),
            v2.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.1
            ),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            v2.RandomErasing(p=0.25)
            ]),
            "clean" : v2.Compose([
            v2.RandomResizedCrop(320, scale=(0.6,1.0), antialias=True),
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(p=0.3),
            v2.RandomRotation(15),
            v2.ColorJitter(0.2, 0.2, 0.2),
            # Extra studio augmentations before normalize
            v2.GaussianBlur(3, (0.1, 1.0)),
            #v2.RandomPerspective(0.1, 0.3),
            v2.RandomAutocontrast(p=0.1),
            v2.RandomGrayscale(p=0.1),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            v2.RandomErasing(p=0.3)
            ])
            
        }            

        train_transform_4 = {
            "standard" : v2.Compose([
            #v2.Resize((256, 256)),
            #v2.PILToTensor(),
            v2.RandomResizedCrop(320, scale=(0.7,1.0),antialias=True),
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(p=0.2),
            v2.RandomRotation(8),
            v2.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.1
            ),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            v2.RandomErasing(p=0.1)
            ]),
            "clean" : v2.Compose([
            v2.RandomResizedCrop(320, scale=(0.7,1.0), antialias=True),
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(p=0.2),
            v2.RandomRotation(10),
            v2.ColorJitter(0.15, 0.15, 0.15),
            # Extra studio augmentations before normalize
            v2.GaussianBlur(3, (0.1, 0.5)),
            #v2.RandomPerspective(0.1, 0.2),
            v2.RandomAutocontrast(p=0.1),
            v2.RandomGrayscale(p=0.1),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            v2.RandomErasing(p=0.2)
            ])
            
        }                

        val_transform_1 = {
            "default" : v2.Compose([
            v2.Resize((224, 224)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
            ])
        }
            
        val_transform_2 = {
            "default" : v2.Compose([
            v2.Resize((256, 256)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
            ])
        }

        val_transform_3= {
            "default" : v2.Compose([
            v2.Resize((320, 320)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
            ])
        }


        # Creating datasets - Stage 2 
        train_ds_stage2 = CustomWasteData(
            stg2_meta_df,
            split="train",
            label_col="stage2_label",
            transform_dict=train_transform_1,
            train=True
        )

        val_ds_stage2 = CustomWasteData(
            stg2_meta_df,
            split="val",
            label_col="stage2_label",
            transform_dict=val_transform_1,
            train=False
        )
        
        test_ds_stage2 = CustomWasteData(
            stg2_meta_df,
            split="test",
            label_col="stage2_label",
            transform_dict=val_transform_3,
            train=False
        )
        
        #WeightedRandomSampler 
        #Retrieving integer class label for every sample
        class_labels = train_ds_stage2.targets
        class_counts = Counter(class_labels)
        num_classes = len(class_counts)
        #Dynamic counts list
        counts_list = [class_counts[i] for i in range(num_classes)]
        #Class weights as inverse frequency
        class_weights = 1.0 / torch.tensor(counts_list, dtype=torch.float)
        #Assign per-sample weight based on class weights
        sample_weights = [class_weights[label] for label in class_labels]
        sample_weights = torch.tensor(sample_weights, dtype=torch.float)
        #WeightedRandomSampler with replacement true to balance minority and majority class samples
        w_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

        num_classes = 13
        #Normalized class weights for use in loss function
        #Normalization done to avoid extremely low class weight values affecting loss
        class_weights_norm = class_weights * num_classes / class_weights.sum()

        #Dataloader setup for Train/Val/Test

        train_loader_s2 = DataLoader(
            train_ds_stage2,
            batch_size=32,
            sampler=w_sampler,
            num_workers=4,
            pin_memory=True,
            persistent_workers=False
        )

        val_loader_s2 = DataLoader(
            val_ds_stage2,
            batch_size=32,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=False
        )

        test_loader_s2 = DataLoader(
            test_ds_stage2,
            batch_size=32,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=False
        )
        class_names = val_ds_stage2.classes
        
        # Resnet34
        print('--------------------------------')
        print("Resnet34 Model eval and training")
        print('--------------------------------')
        res_model = CNN_stage2_r34(13).to(device)
        #Train model on train plus epoch wise evaluation on val set
        train_time, best_model_file= train_model(res_model,train_loader_s2,val_loader_s2,device,output_dir,train_ds_stage2,val_ds_stage2,
                                train_transform_1,train_transform_2,train_transform_3,train_transform_4,val_transform_1,
                                val_transform_2,val_transform_3,class_weights_norm)
        #Final evaluation on test set
        eval_model(res_model,best_model_file,test_loader_s2,device,output_dir,class_names,train_time,val_file,test_file)
        
        print('-------------------------------')
        print("MobilenetV2 Model eval and training")
        print('-------------------------------')
        # MobilenetV2 
        mnv2_model = CNN_stage2_mnv2(13).to(device)
        train_time, best_model_file = train_model(mnv2_model,train_loader_s2,val_loader_s2,device,output_dir,train_ds_stage2,val_ds_stage2,
                                train_transform_1,train_transform_2,train_transform_3,train_transform_4,val_transform_1,
                                val_transform_2,val_transform_3,class_weights_norm)
        eval_model(mnv2_model,best_model_file,test_loader_s2,device,output_dir,class_names,train_time,val_file,test_file)
        
        print('-----------------------------------')
        print("Densenet121 Model eval and training")
        print('-----------------------------------')
        # Densenet121
        dn121_model = CNN_stage2_dn121(13).to(device)
        
        train_time, best_model_file = train_model(dn121_model,train_loader_s2,val_loader_s2,device,output_dir,train_ds_stage2,val_ds_stage2,
                                train_transform_1,train_transform_2,train_transform_3,train_transform_4,val_transform_1,
                                val_transform_2,val_transform_3,class_weights_norm)
        eval_model(dn121_model,best_model_file,test_loader_s2,device,output_dir,class_names,train_time,val_file,test_file)

if __name__ == "__main__":
    main()