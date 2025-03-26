import os
from ultralytics import YOLOE
from pathlib import Path
import argparse
import torch

def yoloe_export_onnx(yoloe_model_path):
    det_model = YOLOE("yoloe-v8m.yaml")
    state = torch.load(yoloe_model_path)

    det_model.load(state["model"])
    det_model.save("yoloe-v8m-seg-det.pt")

    model = YOLOE("yoloe-v8m-seg-det.pt")
    model.eval()

    model.model.model[-1].is_fused = True
    model.model.model[-1].nc = 5

    onnx_path = model.export(format='onnx', opset=13, simplify=True)
  
    os.rename(onnx_path, "yoloe_v8m_det.onnx")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--yoloe_model_path", type=str, help="Input your yoloe model."
    )
    args = parser.parse_args()

    yoloe_model_path = args.yoloe_model_path
    yoloe_export_onnx(yoloe_model_path)



 
