import json
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load class names
with open("class_names.json") as f:
    class_names = json.load(f)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Rebuild ConvNeXt-Tiny architecture
model = models.convnext_tiny(weights=None)
model.classifier[2] = nn.Linear(model.classifier[2].in_features, 102)

# Load saved weights
checkpoint = torch.load("best_flower_model.pth", map_location=device, weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

# Same transform as validation
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    image = Image.open(file.stream).convert("RGB")

    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)

    top5 = torch.topk(probs, 5)
    results = [
        {
            "name": class_names[str(i.item())],
            "confidence": round(float(p) * 100, 2),
        }
        for p, i in zip(top5.values[0], top5.indices[0])
    ]

    return jsonify({"predictions": results})


if __name__ == "__main__":
    app.run(debug=True, port=7860)
