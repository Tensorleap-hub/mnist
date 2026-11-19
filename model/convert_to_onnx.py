"""Export the Keras .h5 model to ONNX and check the two agree.

The Tensorleap engine runs ONNX models only. Usage (Python 3.10, outside the platform):
    pip install tensorflow==2.13.0 tf2onnx==1.16.1 onnxruntime==1.18.1 "numpy<1.25" "protobuf<4"
    python convert_to_onnx.py <model.h5> <model.onnx>
The batch dimension is exported as dynamic.
"""
import sys

import numpy as np
import onnxruntime as ort
import tensorflow as tf
import tf2onnx

src, dst = sys.argv[1], sys.argv[2]
model = tf.keras.models.load_model(src, compile=False)
spec = [tf.TensorSpec((None,) + tuple(i.shape[1:]), i.dtype, name=i.name.split(":")[0]) for i in model.inputs]
tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path=dst)

sess = ort.InferenceSession(dst, providers=["CPUExecutionProvider"])
rng = np.random.default_rng(0)
for batch in (1, 7):
    feeds = [rng.random((batch,) + tuple(i.shape[1:]), dtype=np.float32) for i in model.inputs]
    want = model(feeds if len(feeds) > 1 else feeds[0], training=False)
    want = [np.asarray(w) for w in (want if isinstance(want, (list, tuple)) else [want])]
    got = sess.run(None, {i.name: f for i, f in zip(sess.get_inputs(), feeds)})
    diff = max(float(np.max(np.abs(w - g))) for w, g in zip(want, got))
    assert diff < 1e-4, f"batch {batch}: Keras and ONNX differ by {diff}"
    print(f"batch {batch}: max |keras - onnx| = {diff:.1e}")
print("wrote", dst)
