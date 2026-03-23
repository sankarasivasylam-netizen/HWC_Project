import torch
from torchvision import models
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from pathlib import Path
import os

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
        self.network.classifier[1] = torch.nn.Linear(num_features, 2)

    def forward(self, x):
        return self.network(x)

    def freeze(self):

        # Freeze all layers
        for param in self.network.parameters():
            param.requires_grad = False

        # Unfreeze last layer and 1 before
        for param in self.network.features[-2:].parameters():
            param.requires_grad = True

        # Unfreeze final classifier layer
        for param in self.network.classifier.parameters():
            param.requires_grad = True

    def unfreeze(self):

        for param in self.network.parameters():
            param.requires_grad = True

def load_model():
    model = CNN_stage1_mnv2(2).to(device)  # replace with YOUR architecture
    #model.fc = torch.nn.Linear(model.fc.in_features, 2)
    
    #os.chdir("..")
    
    BASE_DIR = Path(__file__).resolve().parent   # folder of this file

    #MODEL_PATH = BASE_DIR.parent / "model" / "best_model.pth"
    
    #BASE_DIR = Path.cwd()
    pth_dir = BASE_DIR.parent / "output" / "Stage1" / "pretrained" / "V5"
    
    
    mnv2_files = [
	os.path.join(pth_dir, f)
	for f in os.listdir(pth_dir)
        if "mnv2" in f and f.endswith(".pth")
	]

    mnv2_pth = max(mnv2_files, key=os.path.getmtime)

    mnv2_ckpt = torch.load(pth_dir/mnv2_pth, map_location=device)
    model.load_state_dict(mnv2_ckpt["model_state_dict"])
	
    #model.load_state_dict(torch.load(pth_dir/mnv2_pth, map_location=device))
    model.to(device)
    model.eval()

    return model

model = load_model()