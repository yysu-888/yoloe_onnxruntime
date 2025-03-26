import torch
import torch.nn as nn
import onnxruntime as ort
import cv2
import numpy as np
import argparse
import open_clip

class YoloeOnnx:
    def __init__(
        self,
        mobile_clip_model_path,
        yoloe_model_path,
        confidence_thres=0.25,
        iou_thres=0.7,
        input_height=640,
        input_width=640,
    ):
        self.input_height = input_height
        self.input_width = input_width
        self.max_len = 5
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self.mobile_clip_session = ort.InferenceSession(
            mobile_clip_model_path, providers=providers
        )
        self.yoloe_model_session = ort.InferenceSession(
            yoloe_model_path, providers=providers
        )

        self.tokenizer = open_clip.get_tokenizer("ViT-B-16")

    def get_tpe(self, text):
        idx = self.tokenizer(text).cpu().numpy()
        model_inputs = self.mobile_clip_session.get_inputs()
        print(idx.shape)
        outputs = self.mobile_clip_session.run(None, {model_inputs[0].name: idx})
        return outputs[0]
       
    def process(self, img_ori, text):
        tpe = self.get_tpe(text)
        tpe = np.expand_dims(tpe, 0)

        assert self.max_len - len(text) > 0, "text support max length:{}".format(
            self.max_len
        )
        tpe_dummpy = np.zeros(
            (1, self.max_len - len(text), tpe.shape[2]), dtype=np.float32
        )

        tpe = np.concatenate([tpe, tpe_dummpy], axis=1).astype(np.float32)

        img_process, ratio, (pad_w, pad_h) = self.preprocess(img_ori)
        res = self.yoloe_model_session.run(None, {"x": img_process, "tpe": tpe})[0]
        det_bbx = self.postprocess(img_ori, res, pad_w, pad_h, ratio)
        return det_bbx

    def preprocess(self, img):
        shape = img.shape[:2]
        new_shape = (self.input_height, self.input_width)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        ratio = r, r
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        pad_w, pad_h = (new_shape[1] - new_unpad[0]) / 2, (
            new_shape[0] - new_unpad[1]
        ) / 2
        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
        left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )

        img = (
            np.ascontiguousarray(np.einsum("HWC->CHW", img)[::-1], dtype=np.float32)
            / 255.0
        )
        img_process = img[None] if len(img.shape) == 3 else img
        return img_process, ratio, (pad_w, pad_h)

    def postprocess(self, image_ori, x, pad_w, pad_h, ratio):
        x = np.einsum("bcn->bnc", x)
        x = x[np.amax(x[..., 4:], axis=-1) > self.confidence_thres]
        x = np.c_[
            x[..., :4], np.amax(x[..., 4:], axis=-1), np.argmax(x[..., 4:], axis=-1)
        ]
        x = x[
            cv2.dnn.NMSBoxes(x[:, :4], x[:, 4], self.confidence_thres, self.iou_thres)
        ]
        if len(x) > 0:
            x[..., [0, 1]] -= x[..., [2, 3]] / 2
            x[..., [2, 3]] += x[..., [0, 1]]

            x[..., :4] -= [pad_w, pad_h, pad_w, pad_h]
            x[..., :4] /= min(ratio)

            x[..., [0, 2]] = x[:, [0, 2]].clip(0, image_ori.shape[1])
            x[..., [1, 3]] = x[:, [1, 3]].clip(0, image_ori.shape[0])
            x[..., [0, 1]]
        return x

    def draw_detections(self, img, box, text):
        for i in range(box.shape[0]):
            x1, y1, x2, y2, score, idx = box[i]
            color = [
                (255, 0, 255),
                (0, 255, 255),
                (255, 165, 0),
                (128, 0, 128),
                (0, 128, 128),
                (255, 192, 203),
                (255, 0, 0),
                (0, 255, 0),
                (0, 0, 255),
                (255, 255, 0),
            ]
            cv2.rectangle(
                img, (int(x1), int(y1)), (int(x2), int(y2)), color[int(idx)], 2
            )

            label = f"{text[int(idx)]}: {score:.3f}"

            (label_width, label_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )

            label_x = int(x1)
            label_y = int(y1 - 10) if y1 - 10 > label_height else int(y1 + 10)

            cv2.rectangle(
                img,
                (label_x, label_y - label_height),
                (label_x + label_width, label_y + label_height),
                color[int(idx)],
                cv2.FILLED,
            )

            cv2.putText(
                img,
                label,
                (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        return img


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mobile_clip_onnx_path", type=str, help="Input your ONNX model."
    )
    parser.add_argument("--yoloe_onnx_path", type=str, help="Input your ONNX model.")
    parser.add_argument("--img_path", type=str, help="Path to input image.")
    parser.add_argument("--text_prompt", type=str, help="text prompt(seperate ,)")
    parser.add_argument(
        "--conf_thres", type=float, default=0.25, help="Confidence threshold"
    )
    parser.add_argument(
        "--iou_thres", type=float, default=0.7, help="NMS IoU threshold"
    )

    args = parser.parse_args()
    mobile_clip_onnx_path = args.mobile_clip_onnx_path
    yoloe_onnx_path = args.yoloe_onnx_path
    confidence_thres = args.conf_thres
    iou_thres = args.iou_thres
    img_path = args.img_path
    text = args.text_prompt
    text = text.strip().split(",")

    yoloe = YoloeOnnx(
        mobile_clip_onnx_path, yoloe_onnx_path, confidence_thres, iou_thres
    )
    img_ori = cv2.imread(img_path)
    det_bbx = yoloe.process(img_ori, text)
    img_viz = yoloe.draw_detections(img_ori, det_bbx, text)
    cv2.imwrite("result.jpg", img_viz)



