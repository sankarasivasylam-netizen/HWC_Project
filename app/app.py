import torch
import gradio as gr
from PIL import Image
import torchvision.transforms as transforms
import time
from model_loader import get_stg1_model,get_stg2_model,device,get_stg2_model_cam
from utils import CLASS_NAMES, CLASS_NAMES_STG2
import io
from huggingface_hub import hf_hub_download
import numpy as np
import cv2
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import ImageOps
from torchvision.transforms import v2
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


transform_stg1 = v2.Compose([
                v2.Resize((256, 256)),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
                ])

transform_stg2 = v2.Compose([
            v2.Resize((320,320)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
            ])

cam_stage_2 = None

def classify_image(image):

    start = time.time()
    
    image = image.convert("RGB")

    tensor = transform_stg1(image).unsqueeze(0).to(device)

    model_stg1 = get_stg1_model()
    model_stg1.eval()

    with torch.no_grad():

        output = model_stg1(tensor)
        probs = torch.softmax(output, dim=1)
        confidence, pred_class = torch.max(probs, dim=1)
    label_stg1 = CLASS_NAMES[pred_class.item()]
    conf_stg1 = confidence.item()

    if label_stg1 == "Non-Organic":

        model_stg2 = get_stg2_model()
        model_stg2.eval()
        tensor_2 = transform_stg2(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            
            output_2 = model_stg2(tensor_2)

            probs_2 = torch.softmax(output_2, dim=1)
            confidence_2, pred_class_2 = torch.max(probs_2, dim=1)

        label_stg2 = CLASS_NAMES_STG2[pred_class_2.item()]
        conf_stg2 = confidence_2.item()
        
        stage1_scores = {
        "Organic":     conf_stg1 if pred_class == 1 else 1 - conf_stg1,
        "Non_Organic": conf_stg1 if pred_class == 0 else 1 - conf_stg1,
        }

        stg2_classes = list(CLASS_NAMES_STG2.values())
        stage2_scores = {cls: prob for cls, prob in zip(stg2_classes, probs_2.squeeze().tolist())}
        info_txt = None
        
    
        

    else:

        stage1_scores = {
        "Organic":     conf_stg1 if pred_class == 1 else 1 - conf_stg1,
        "Non_Organic": conf_stg1 if pred_class == 0 else 1 - conf_stg1,
        }

        stage2_scores =  None
        info_txt = "Organic waste — no stage-2 classification"
        
        
    end = time.time()

    inference_txt = f"⏱ Inference time: {(end-start)*1000:.1f} ms"
    
    return stage1_scores,stage2_scores,inference_txt,info_txt,gr.update(interactive=True)

def get_stage2_cam(model_stg2_gc):
    
    global cam_stage_2

    if cam_stage_2 is None:

        target_layer_2 = model_stg2_gc.network.features.denseblock4
        cam_stage_2 = GradCAMPlusPlus(
            model=model_stg2_gc,
            target_layers=[target_layer_2]
        )

    return cam_stage_2

def explain_image(image):

    image = image.convert("RGB")
    input_tensor_1 = transform_stg1(image).unsqueeze(0).to(device)
    img_np_1 = np.array(image.resize((256,256))) / 255.0

    model_stg1 = get_stg1_model()
    target_layer_1 = model_stg1.network.features[-1]

    with torch.no_grad():
        output = model_stg1(input_tensor_1)
    
        probs = torch.softmax(output, dim=1)
        conf, pred = torch.max(probs, 1)
    
    targets = [ClassifierOutputTarget(pred.item())]

                
    cam_1 = GradCAMPlusPlus(model=model_stg1, target_layers=[target_layer_1])
    grayscale_cam_1 = cam_1(input_tensor=input_tensor_1,targets=targets)[0]
    vis_1 = show_cam_on_image(img_np_1, grayscale_cam_1, use_rgb=True)

    label_stg1 = CLASS_NAMES[pred.item()]

    if label_stg1 == "Non-Organic":

        model_stg2_gc = get_stg2_model_cam()
        model_stg2_gc.eval()
        
        cam_2 = None
        vis_2 = None

        input_tensor_2 = transform_stg2(image).unsqueeze(0).to(device)
        img_np_2 = np.array(image.resize((320,320))) / 255.0
        cam_stage2_local = get_stage2_cam(model_stg2_gc)

        with torch.no_grad():
            output = model_stg2_gc(input_tensor_2)
        
            probs = torch.softmax(output, dim=1)
            conf, pred = torch.max(probs, 1)
        
        targets = [ClassifierOutputTarget(pred.item())]
        
        grayscale_cam_2 = cam_stage2_local(input_tensor=input_tensor_2,targets=targets)[0]
        vis_2 = show_cam_on_image(img_np_2, grayscale_cam_2, use_rgb=True)

        return vis_1, vis_2
    else:
        return vis_1, None

with gr.Blocks() as wc:

    title = gr.HTML("<h1 style='text-align:center;'>♻️ Hierarchical Waste Classifier</h1>")
    
    gr.Markdown("""
    ⬆️ Upload an image to classify waste:

    1️⃣ Stage 1 → Organic vs Non-Organic   2️⃣ Stage 2 → Detailed classification      3️⃣ Explain to view Grad-CAM  
    """)


    with gr.Row():

        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Upload Image",height=320,width=380)

            with gr.Row():
                predict_btn = gr.Button("🚀 Predict",variant = 'primary',interactive=False)
                explain_btn = gr.Button("🧠 Explain",variant = 'secondary',interactive=False)
                clear_btn = gr.Button("🗑️️ Clear",variant = 'stop')
            with gr.Row():
                cam1_output = gr.Image(label="Stage-1 GradCAM MobilenetV2",height=320,width=320)
                cam2_output = gr.Image(label="Stage-2 GradCAM Densenet121",height=320,width=320)

            image_input.change(
                               fn=lambda img: gr.update(interactive=img is not None),
                               inputs=image_input,
                               outputs=predict_btn
                              )

        with gr.Column(scale=1):
            stg1_out  = gr.Label(label="Stage 1 - Organic/Non-Organic", num_top_classes=2)
            stg2_out  = gr.Label(label="Stage 2 - Non-Organic Category", num_top_classes=5) 
            time_out  = gr.Textbox(label="⌛ Performance", interactive=False)
            msg_out   = gr.Textbox(label="ℹ️ Info", interactive=False)

            
    predict_btn.click(
        fn=classify_image,
        inputs=image_input,
        outputs=[stg1_out, stg2_out, time_out,msg_out,explain_btn]
    )

    explain_btn.click(
        fn=explain_image,
        inputs=image_input,
        outputs=[cam1_output, cam2_output]
    )

    clear_btn.click(
        fn=lambda: (None, {}, {}, "", "", None, None,gr.update(interactive=False),gr.update(interactive=False)),
        inputs=None,
        outputs=[image_input, stg1_out, stg2_out, time_out,msg_out , cam1_output, cam2_output,predict_btn, explain_btn]
    )

wc.launch(theme=gr.themes.Default(),share=False)
