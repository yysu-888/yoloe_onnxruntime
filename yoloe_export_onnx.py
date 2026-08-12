#!/usr/bin/env python3
"""
Correctly exports YOLOE models to ONNX format.

Clarification on `yoloe-v8m.yaml`:
The `.yaml` file only defines the model architecture. The actual weights come from the `.pt` file.
Converting `yoloe-v8m.yaml` to a `.pt` file does not automatically include trained weights.
You must load the pretrained `.pt` model first, then export it to ONNX.
"""
from ultralytics import YOLOE


def export_onnx(model_id: str = "yoloe-v8m") -> None:
    """Export a YOLOE model to ONNX format using pretrained weights."""
    # Load the pretrained model directly. YOLOE automatically loads the correct YAML from the .pt file.
    model = YOLOE(f"{model_id}-seg.pt")

    # Export to ONNX. Setting simplify=False is recommended for YOLOE to preserve custom heads.
    model.export(format="onnx", simplify=False, imgsz=640)
    print(f"Successfully exported {model_id} to ONNX.")


if __name__ == "__main__":
    import sys
    model_id = sys.argv[1] if len(sys.argv) > 1 else "yoloe-v8m"
    export_onnx(model_id)
