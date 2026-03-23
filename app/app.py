from fastapi import FastAPI, File, UploadFile, Form
from PIL import Image
import io
import torch
import time
from torchvision import transforms
from model_loader import model, device
from utils import CLASS_NAMES
from fastapi.responses import HTMLResponse

app = FastAPI()

transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
            ])
	    

@app.post("/predict")
async def predict(
        file: UploadFile = File(...),
        true_label: int = Form(None)   # optional
    ):
    
    start = time.time()

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGBA").convert("RGB")

    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)

        confidence, pred_class = torch.max(probs, dim=1)

    inference_time = time.time() - start

    response = {
        "predicted_class_id": int(pred_class.item()),
        "predicted_label": CLASS_NAMES[int(pred_class.item())],
        "confidence": float(f"{confidence.item():.2f}"),
        "all_probabilities": probs.cpu().numpy().tolist()[0],
        "inference_time_sec": float(f"{inference_time:.2f}"),
        "model_version": model.__class__.__name__
    }

    # If true label provided → evaluate
    if true_label is not None:
        response["true_class_id"] = true_label
        response["true_label"] = CLASS_NAMES[true_label]
        response["correct"] = bool(true_label == pred_class.item())

    return response

@app.get("/ui", response_class=HTMLResponse)
def ui():
    return """
    <html>
    <body>

        <h2>Image Prediction</h2>

        <input id="file" type="file" onchange="preview()">
        <br><br>

        <img id="preview_img" width="300"/>
        <br><br>

        <button onclick="upload()">Predict</button>

        <h3 id="result"></h3>

        <script>

        function preview(){
            let file = document.getElementById("file").files[0];
            let reader = new FileReader();

            reader.onload = function(e){
                document.getElementById("preview_img").src = e.target.result;
            }

            reader.readAsDataURL(file);
        }

        async function upload() {

            let fileInput = document.getElementById("file");

            let formData = new FormData();
            formData.append("file", fileInput.files[0]);

            let response = await fetch("/predict", {
                method: "POST",
                body: formData
            });

            let data = await response.json();

            document.getElementById("result").innerHTML =
                "Prediction: " + data.predicted_label +
                "<br>Confidence: " + data.confidence +
		"<br> Inference time: " + data.inference_time_sec ;
        }

        </script>

    </body>
    </html>
    """
    
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)