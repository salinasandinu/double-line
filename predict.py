"""
Simple prediction script for double white line detection.
Usage: python predict.py --image path/to/image.jpg
"""

from ultralytics import YOLO
import argparse
import os


def predict(model_path, source, conf=0.25, save=True, show=False):
    """
    Run prediction on an image, video, or folder.
    
    Args:
        model_path: Path to the trained model (.pt file)
        source: Image/video path, folder, or 0 for webcam
        conf: Confidence threshold (0-1)
        save: Save results to disk
        show: Display results in window
    """
    # Load model
    model = YOLO(model_path)
    
    # Run prediction
    results = model.predict(
        source=source,
        conf=conf,
        save=save,
        save_txt=save,
        project="D:/Academics/Research/double-line/runs/predict",
        name="results",
        exist_ok=True,
        show=show,
        line_width=2,
    )
    
    # Print detection summary
    for r in results:
        boxes = r.boxes
        if len(boxes) > 0:
            print(f"\nDetected {len(boxes)} double white line(s) in: {r.path}")
            for box in boxes:
                conf_val = box.conf[0].item()
                print(f"  - Confidence: {conf_val:.2%}")
        else:
            print(f"\nNo detections in: {r.path}")
    
    if save:
        print(f"\nResults saved to: D:/Academics/Research/double-line/runs/predict/results")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Double White Line Detection")
    parser.add_argument("--model", type=str, 
                        default="C:/Users/Default/Downloads/bestg.pt",
                        help="Path to model weights (.pt file)")
    parser.add_argument("--source", type=str, required=True,
                        help="Image/video path, folder, or 0 for webcam")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold (0-1)")
    parser.add_argument("--no-save", action="store_true",
                        help="Don't save results")
    parser.add_argument("--show", action="store_true",
                        help="Display results in window")
    
    args = parser.parse_args()
    
    # Check if source exists (unless it's webcam)
    if args.source != "0" and not os.path.exists(args.source):
        print(f"Error: Source not found: {args.source}")
        exit(1)
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model not found: {args.model}")
        print("Please provide the correct path to your model with --model")
        exit(1)
    
    predict(
        model_path=args.model,
        source=args.source,
        conf=args.conf,
        save=not args.no_save,
        show=args.show
    )
