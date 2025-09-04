#!/usr/bin/env python3
import torch
import argparse
from pathlib import Path
from basicsr.archs.rrdbnet_arch import RRDBNet

parser = argparse.ArgumentParser()
parser.add_argument('--weights', required=True)
parser.add_argument('--output', default='realesrgan_x4.onnx')
parser.add_argument('--opset', type=int, default=11)
parser.add_argument('--size', type=int, default=128)
args = parser.parse_args()

pth = args.weights
output = args.output
opset = args.opset
size = args.size

print('Loading model...')
net = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23)
state = torch.load(pth, map_location='cpu')
try:
    net.load_state_dict(state, strict=False)
except Exception as e:
    if 'params' in state:
        net.load_state_dict(state['params'], strict=False)
    else:
        print('Warning: load_state_dict failed:', e)
net.eval()

dummy = torch.randn(1,3,size,size)
print('Exporting to ONNX...')
torch.onnx.export(net, dummy, output, opset_version=opset, input_names=['input'], output_names=['output'],
                  dynamic_axes={'input':{0:'batch',2:'h',3:'w'}, 'output':{0:'batch',2:'h',3:'w'}})
print('Saved', output)
