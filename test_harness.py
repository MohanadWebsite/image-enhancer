#!/usr/bin/env python3
# Simple test harness: runs an ONNX model on an input image and saves output.
import onnxruntime as ort
from PIL import Image
import numpy as np
import argparse, os, time

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True)
parser.add_argument('--image', required=True)
parser.add_argument('--out', default='out.png')
args = parser.parse_args()

sess = ort.InferenceSession(args.model, providers=['CPUExecutionProvider'])
img = Image.open(args.image).convert('RGB')
w,h = img.size
# resize small for test
img_small = img.resize((w,w))
arr = np.array(img_small).astype('float32')/255.0
# HWC -> CHW
arr = np.transpose(arr, (2,0,1))
arr = np.expand_dims(arr, 0)
inputs = {sess.get_inputs()[0].name: arr}
t0 = time.time()
out = sess.run(None, inputs)
t1 = time.time()
print('Inference time:', t1-t0)
# Attempt to convert output to image (assume output is [1,3,H,W])
out_arr = out[0]
if isinstance(out_arr, np.ndarray):
    if out_arr.ndim == 4:
        out_img = np.squeeze(out_arr, 0)
        out_img = np.transpose(out_img, (1,2,0))
        out_img = np.clip(out_img*255.0, 0, 255).astype('uint8')
        Image.fromarray(out_img).save(args.out)
        print('Saved output to', args.out)
    else:
        print('Unexpected output shape', out_arr.shape)
else:
    print('Output is not numpy array')
