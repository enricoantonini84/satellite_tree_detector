import torch
from ultralytics import YOLO
import argparse

#ok, this is HEAVVILY inspired by yolo repo, thanks Ultralytics!

#this is the real deal
def trainYOLO11(basePath, datasetYamlPath):
    if torch.backends.mps.is_available():
        device = "mps" #mixed results, it'll turn your mac into a toaster while train
    elif torch.cuda.is_available():
        device = "cuda" #best performances
    else:
        device = "cpu" #very poor performances
    model = YOLO("yolo11x.pt")

    # Training configuration
    train_args = {
        'data': datasetYamlPath,
        'epochs': 300,
        'imgsz': 640,
        'batch': 16,
        'device': device,
        'workers': 8 if device != "mps" else 4,
        'patience': 100,
        'save_period': 10,
        'project': f"{basePath}/runs/detect",
        'name': 'tree_detection',
        'pretrained': True,
        'lr0': 0.01,
        'lrf': 0.1,
        'cos_lr': True,
        'amp': True,
        'cache': "disk",
        'single_cls': True
    }

    print(f"Starting YOLO11x training on {device}...")
    print(f"With dataset: {datasetYamlPath}")

    results = model.train(**train_args)
    metrics = model.val()

    print(f"Training completed!")
    print(f"Best model saved at: weights/best.pt")
    print(f"Validation mAP50: {metrics.box.map50:.3f}")
    print(f"Validation mAP50-95: {metrics.box.map:.3f}")

    return model, results, metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--basePath", type=str, required=True, help="Base path for the dataset and output")
    parser.add_argument("--datasetYamlPath", type=str, required=True, help="Path to the dataset YAML file")
    
    args = parser.parse_args()
    
    try:
        print("Starting YOLO11x Tree Cover Detection Training")

        model, results, metrics = trainYOLO11(args.basePath, args.datasetYamlPath)

        print(f"Training Summary:")
        print(f"Device: {torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')}")
        print(f"Final mAP50: {metrics.box.map50:.3f}")
        print(f"Final mAP50-95: {metrics.box.map:.3f}")
        print(f"Best weights: {args.basePath}/runs/detect/tree_detection/weights/best.pt")

    except Exception as e:
        print(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()