## Quick Start

### Installation

<details open>
<summary><strong>Install via pip (Recommended)</strong></summary>

```bash
# 1. Create and activate a new environment
conda create -n yolo-fuadapter python=3.11 -y
conda activate yolo-fuadapter

# 2. Clone the repository
git clone https://github.com/Simpleupper/YOLO-FUAdapter
cd YOLO-FUAdapter

# 3. Install dependencies
pip install -r requirements.txt
pip install -e .

# 4. Optional: Install FlashAttention for faster training (CUDA required)
pip install flash_attn
pip install opencv-python-headless
```
</details>

### Validation

Validate the model accuracy on the NDT dataset.

```python
from ultralytics import YOLO

# Load the pretrained model
model = YOLO("yolo-fuadapter.pt") 

# Run validation
metrics = model.val(data="ndt_dataset.yaml", save_json=True)
print(metrics.box.map)  # map50-95
```

### Training

Train a new model on your custom dataset or NDT.

```python
from ultralytics import YOLO

# Load a model
model = YOLO('ultralytics/cfg/models/11/yolo-fuadapter.yaml')  # build a new model from YAML

# Train the model
results = model.train(
    data='ndt_dataset.yaml',
    epochs=600, 
    batch=64, 
    imgsz=320,
    device="0,1,2,3", # Use multiple GPUs
    scale=0.5, 
    mosaic=1.0,
    mixup=0.0, 
    copy_paste=0.1
)
```

### Inference

Run inference on images or videos.

**Python:**
```python
from ultralytics import YOLO

model = YOLO("yolo-fuadapter.pt")
results = model("path/to/image.jpg")
results[0].show()
```

**CLI:**
```bash
yolo predict model=yolo-fuadapter.pt source='path/to/image.jpg' show=True
```

### Export

Export the model to other formats for deployment (TensorRT, ONNX, etc.).

```python
from ultralytics import YOLO

model = YOLO("yolo-fuadapter.pt")
model.export(format="engine", half=True)  # Export to TensorRT
# formats: onnx, openvino, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs
```

⭐ **If you find this work useful, please star the repository!**
