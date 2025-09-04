#!/usr/bin/env python3
# Placeholder for GFPGAN conversion. Customize according to your GFPGAN repo.
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--weights', required=True)
parser.add_argument('--output', default='gfpgan.onnx')
parser.add_argument('--opset', type=int, default=11)
parser.add_argument('--size', type=int, default=512)
args = parser.parse_args()

print('GFPGAN export: customize this script for your GFPGAN version and run it')
