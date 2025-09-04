#!/usr/bin/env python3
import onnx
from onnxconverter_common.float16 import convert_float_to_float16
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--input', required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()

m = onnx.load(args.input)
m16 = convert_float_to_float16(m)
onnx.save(m16, args.output)
print('Saved FP16 model to', args.output)
