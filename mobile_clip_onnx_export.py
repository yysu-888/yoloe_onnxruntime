import torch
from ultralytics.nn.text_model import build_text_model
import torch.nn as nn
import onnxruntime as ort
import argparse

class MobileClipWarpper(nn.Module):
    def __init__(self,model_path,model_name ="mobileclip:blt"):
        super(MobileClipWarpper,self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.mobile_clip_model = build_text_model(model_name, self.device,model_path)

    def forward(self,ids):
        text_feature = self.mobile_clip_model.encode_text(ids)
        return text_feature
    
def test_clip_onnx(onnx_model,txt_model,text):
    ids = txt_model.mobile_clip_model.tokenize(text).cpu().numpy()
    session = ort.InferenceSession(onnx_model, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    model_inputs = session.get_inputs()
    outputs = session.run(None, {model_inputs[0].name: ids})
    return outputs[0]

if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mobile_clip_path", type=str, help="Input your pt model."
    )

    args = parser.parse_args()
    txt_model = MobileClipWarpper(args.mobile_clip_path)
    input_ids = torch.randint(0,1000,(1,77))
    torch.onnx.export(txt_model,  
                  (input_ids), 
                  "mobileclip.onnx",  
                  opset_version=14,          
                  do_constant_folding=True,  
                  input_names=['input_ids'],   
                  output_names=['output'], 
                  dynamic_axes={'input_ids': {0: 'seq_len'}}
    )







