import argparse
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
import os, sys

def export(model_name: str, max_length: int, out_path: str):
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForMaskedLM.from_pretrained(model_name)
    mdl.eval()
    mdl.to("cpu")

    # Dummy inputs (on cpu)
    sample = tok("hello world", return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = sample["input_ids"].to("cpu")
    attention_mask = sample["attention_mask"].to("cpu")

    # ensure directory exists
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    try:
        with torch.no_grad():
            # Use dynamic_axes for variable batch and sequence length
            dynamic_axes = {
                "input_ids": {0: "batch", 1: "seq"},
                "attention_mask": {0: "batch", 1: "seq"},
                "logits": {0: "batch", 1: "seq"},
            }
            torch.onnx.export(
                mdl,
                (input_ids, attention_mask),
                out_path,
                input_names=["input_ids", "attention_mask"],
                output_names=["logits"],
                opset_version=18,
                do_constant_folding=True,
                dynamic_axes=dynamic_axes,
            )
    except Exception as e:
        print("ERROR: ONNX export failed:", e, file=sys.stderr)
        raise

    # validate the produced model
    onnx_model = onnx.load(out_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX export successful:", out_path)

def quantize(in_path: str, out_path: str):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    quantize_dynamic(in_path, out_path, weight_type=QuantType.QInt8)
    print("Quantized model written to:", out_path)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilbert-base-uncased")
    ap.add_argument("--max_length", type=int, default=64)
    ap.add_argument("--out", default="models/distilbert-base-uncased.onnx")
    ap.add_argument("--quant_out", default="models/distilbert-base-uncased.int8.onnx")
    args = ap.parse_args()

    export(args.model, args.max_length, args.out)
    quantize(args.out, args.quant_out)
    print("Exported:", args.out)
    print("Quantized:", args.quant_out)