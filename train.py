"""
Training script for detecting white double lines on roads using YOLOv8.
Dataset: Custom dataset with 'double_white_line' class in YOLO format.
"""

from ultralytics import YOLO
import os

def main():
    # Path to the dataset configuration
    data_yaml = "dataset/data.yaml"
    
    # Verify dataset exists
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"Dataset config not found: {data_yaml}")
    
    # Load a pretrained YOLOv8 model (nano version for faster training)
    # Options: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
    model = YOLO("yolov8n.pt")  # Using nano model, change to 's', 'm', 'l', or 'x' for larger models
    
    # Train the model
    results = model.train(
        data=data_yaml,
        epochs=100,              # Number of training epochs
        imgsz=640,               # Image size
        batch=16,                # Batch size (reduce if running out of memory)
        patience=20,             # Early stopping patience
        save=True,               # Save checkpoints
        device=0,                # Use GPU (set to 'cpu' if no GPU available)
        workers=4,               # Number of dataloader workers
        project="runs/detect",   # Project directory
        name="double_line_det",  # Experiment name
        exist_ok=True,           # Overwrite existing experiment
        pretrained=True,         # Use pretrained weights
        optimizer="auto",        # Optimizer (auto selects best)
        verbose=True,            # Verbose output
        seed=42,                 # Random seed for reproducibility
        val=True,                # Validate during training
        plots=True,              # Generate training plots
    )
    
    print("\n" + "="*50)
    print("Training completed!")
    print("="*50)
    print(f"\nBest model saved at: {model.trainer.best}")
    print(f"Last model saved at: {model.trainer.last}")
    
    return results


if __name__ == "__main__":
    main()
