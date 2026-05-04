import torch
from torchvision import models
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torchvision.models import densenet121, DenseNet121_Weights
from pathlib import Path
import os
from huggingface_hub import hf_hub_download

model_stg1 = None
model_stg2 = None
model_stg2_gc = None

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
	    
class CNN_stage1_mnv2(torch.nn.Module):
    def __init__(self, num_classes, input_size=(224, 224), channels=3):
        super(CNN_stage1_mnv2, self).__init__()

        self.input_size = input_size
        self.channels = channels

        self.network = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)

        num_features = self.network.classifier[1].in_features
        #self.network.classifier[1] = torch.nn.Linear(num_features, 2)
        self.network.classifier[1]  = torch.nn.Sequential(
                                      #torch.nn.Dropout(p=0.3),
                                      torch.nn.Linear(num_features, 256),
                                      torch.nn.BatchNorm1d(256),
                                      torch.nn.ReLU(),
                                      torch.nn.Dropout(p=0.2),
                                      torch.nn.Linear(256, num_classes)
                                      )

    def forward(self, x):
        return self.network(x)

    def freeze(self):

        # Freeze all layers
        for param in self.network.parameters():
            param.requires_grad = False

        for param in self.network.features[14:].parameters():
            param.requires_grad = True

        # Unfreeze final classifier layer
        for param in self.network.classifier.parameters():
            param.requires_grad = True

    def unfreeze(self):

        for param in self.network.parameters():
            param.requires_grad = True

class CNN_stage2_dn121(torch.nn.Module):
    def __init__(self, num_classes, input_size=(256, 256), channels=3):
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

        # Unfreeze 2nd last layer
        for param in self.network.features.denseblock3.parameters():
            param.requires_grad = True

        # Unfreeze last layer
        for param in self.network.features.denseblock4.parameters():
            param.requires_grad = True  

        # Unfreeze final classifier layer
        for param in self.network.classifier.parameters():
            param.requires_grad = True

    def unfreeze(self):

        for param in self.network.parameters():
            param.requires_grad = True



def get_stg1_model():
    
    global model_stg1

    if model_stg1 is None:
        print("Downloading / Loading Stage-1 model...")

        model_stg1 = CNN_stage1_mnv2(2).to(device)  
        
        MODEL_PATH_STG1 = hf_hub_download(
            repo_id="Sanky1309/Stage1-MNV2-Classifier",
            filename="best_CNN_stage1_mnv2_20260422_124939.pth",
            token=os.getenv("HF_TOKEN")
        )


        mnv2_ckpt = torch.load(MODEL_PATH_STG1, map_location=device, weights_only=False)
        model_stg1.load_state_dict(mnv2_ckpt["model_state_dict"])
            
        model_stg1.to(device)
        model_stg1.eval()

    return model_stg1

def get_stg2_model():
    
    global model_stg2

    if model_stg2 is None:
        print("Downloading / Loading Stage-2 model...")

        #model_stg2 = CNN_stage2_dn121(13).to(device)
        MODEL_PATH_STG2 = hf_hub_download(
            repo_id="Sanky1309/Stage2-DN121-Classifier",
            filename="stage2_dn121.pt",
            token=os.getenv("HF_TOKEN")
        )

        #dn121_ckpt = torch.load(MODEL_PATH_STG2, map_location=device)
        #model_stg2.load_state_dict(dn121_ckpt["model_state_dict"])

        model_stg2 = torch.jit.load(MODEL_PATH_STG2)
            
        #model.load_state_dict(torch.load(pth_dir/mnv2_pth, map_location=device))
        model_stg2.to(device)
        model_stg2.eval()

    return model_stg2

def get_stg2_model_cam():
    
    global model_stg2_gc

    if model_stg2_gc is None:
        print("Downloading / Loading Stage-2 Grad-cam model...")

        model_stg2_gc = CNN_stage2_dn121(13).to(device)
        MODEL_PATH_STG2 = hf_hub_download(
            repo_id="Sanky1309/Stage2-DN121-Classifier",
            filename="best_CNN_stage2_dn121_20260415_102814.pth",
            token=os.getenv("HF_TOKEN")
        )

        dn121_ckpt = torch.load(MODEL_PATH_STG2, map_location=device,weights_only=False)
        model_stg2_gc.load_state_dict(dn121_ckpt["model_state_dict"])

        #model_stg2 = torch.jit.load('stage2_dn121_1.pt')
            
        #model.load_state_dict(torch.load(pth_dir/mnv2_pth, map_location=device))
        model_stg2_gc.to(device)
        model_stg2_gc.eval()

    return model_stg2_gc
