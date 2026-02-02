"""
Inference script for detecting white double lines on roads.
Use this script to run predictions on new images/videos after training.
"""

from ultralytics import YOLO
import argparse
import os


def run_inference(model_path, source, save_dir="D:/Academics/Research/double-line/runs/predict", conf=0.25, show=False):
    """
    Run inference using a trained YOLOv8 model.
    
    Args:
        model_path: Path to the trained model weights (.pt file)
        source: Path to image, video, directory, or webcam (0)
        save_dir: Directory to save results
        conf: Confidence threshold
        show: Whether to display results
    """
    # Load the trained model
    model = YOLO(model_path)
    
    # Run inference
    results = model.predict(
        source=source,
        conf=conf,               # Confidence threshold
        iou=0.45,                # NMS IoU threshold
        save=True,               # Save annotated images
        save_txt=True,           # Save results as .txt files
        save_conf=True,          # Include confidence in .txt files
        project=save_dir,        # Save directory
        name="double_line",      # Experiment name
        exist_ok=True,           # Overwrite existing
        show=show,               # Display results
        line_width=2,            # Bounding box line width
    )
    
    print(f"\nResults saved to: {save_dir}/double_line")
    return results


def validate_model(model_path, data_yaml="dataset/data.yaml"):
    """
    Validate the model on the test/validation set.
    
    Args:
        model_path: Path to the trained model weights
        data_yaml: Path to data configuration file
    """
    model = YOLO(model_path)
    
    metrics = model.val(
        data=data_yaml,
        split="test",            # Use test set for validation
        conf=0.25,
        iou=0.5,
        save_json=True,          # Save results in COCO format
        plots=True,              # Generate validation plots
    )
    
    print("\n" + "="*50)
    print("Validation Results")
    print("="*50)
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")
    
    return metrics


def export_model(model_path, format="onnx"):
    """
    Export the trained model to different formats.
    
    Args:
        model_path: Path to the trained model weights
        format: Export format (onnx, torchscript, tflite, etc.)
    """
    model = YOLO(model_path)
    
    # Export to specified format
    model.export(format=format)
    print(f"\nModel exported to {format} format")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Double White Line Detection Inference")
    parser.add_argument("--model", type=str, default="runs/detect/double_line_det/weights/best.pt",
                        help="Path to trained model weights")
    parser.add_argument("--source", type=str, default="dataset/test/images",
                        help="Image/video source (path, directory, or 0 for webcam)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold")
    parser.add_argument("--mode", type=str, choices=["predict", "validate", "export"],
                        default="predict", help="Mode: predict, validate, or export")
    parser.add_argument("--export-format", type=str, default="onnx",
                        help="Export format (onnx, torchscript, tflite)")
    parser.add_argument("--show", action="store_true",
                        help="Display results during inference")
    
    args = parser.parse_args()
    
    if args.mode == "predict":
        run_inference(args.model, args.source, conf=args.conf, show=args.show)
    elif args.mode == "validate":
        validate_model(args.model)
    elif args.mode == "export":
        export_model(args.model, format=args.export_format)
