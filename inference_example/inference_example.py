import onnxruntime as ort
from pathlib import Path
import numpy as np
from tensorflow.keras.losses import CategoricalCrossentropy
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Softmax
import json

def plot_input(img):
    plt.imshow((img[...,0]*255).astype(np.uint8))

def plot_logits(output):
    label_list = [f'{i}-digit' for i in range(10)]
    plt.bar(label_list, Softmax()(output[0]))

#load files from disk
file_name = 'input.npy'
input_arr = np.load(f'data/{file_name}')
gt = np.load('data/gt.npy')
expected_output = np.load('data/output.npy')

# prepare input and gt for inference (normalize if needed)
gt_with_batch = gt[None, ...]
input_with_batch = input_arr[..., 0][None, None, ...]

#load model
onnx_path = Path.cwd().parent / 'model' / 'mnist_onnx.onnx'
ort_session = ort.InferenceSession(onnx_path)
onnx_outputs = ort_session.run(None, {'Input3':input_arr[..., 0][None, None, ...]})
output_vec = onnx_outputs[0]

# verify output
assert  (output_vec - expected_output).sum() < 1e-4

# Compute loss/metrics or any other metric that is important for you
loss_fn = CategoricalCrossentropy(from_logits=True)
loss = loss_fn(gt_with_batch, output_vec)

# Plot inputs and outputs
plot_input(input_arr)
plot_logits(output_vec)

# Load metadata for file
with open('data/metadata.json') as f:
    sample_metadata_dict = json.load(f)

# Print all metadata
sample_metadata = sample_metadata_dict[file_name]
print(sample_metadata)