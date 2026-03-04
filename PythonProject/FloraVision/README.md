# FloraVision — Flower Species Classifier

A deep learning project that classifies **102 flower species** from the [Oxford 102 Flowers](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/) dataset using **ConvNeXt-Tiny** fine-tuned with **Optuna** hyperparameter optimization.

## Live Demo

[huggingface.co/spaces/welyty/FloraVision](https://huggingface.co/spaces/welyty/FloraVision)

## Project Structure

```
FloraVision/
├── app/
│   ├── app.py              # Flask web app
│   ├── class_names.json    # 102 flower class labels
│   ├── requirements.txt    # Dependencies
│   └── templates/
│       └── index.html      # Frontend UI
├── notebook/
│   └── flower_classifier.ipynb   # Training & evaluation
├── images/
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   ├── optimization_history.png
│   ├── param_importances.png
│   └── predictions.png
└── README.md
```

## Model

| Detail | Value |
|---|---|
| Architecture | ConvNeXt-Tiny |
| Dataset | Oxford 102 Flowers |
| Classes | 102 |
| Optimizer | AdamW + Optuna HPO |
| Framework | PyTorch |

## Run Locally

```bash
cd app
pip install -r requirements.txt
# Place best_flower_model.pth in app/
python app.py
# → http://127.0.0.1:7860
```

> The model weights (`best_flower_model.pth`, ~106 MB) are hosted on [Hugging Face](https://huggingface.co/spaces/welyty/FloraVision/resolve/main/best_flower_model.pth) and are not tracked in this repo.
