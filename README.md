# GraphEnet-v2

This repository contains the code for the GraphEnet-v2 model, an advanced graph neural network architecture designed for human pose estimation tasks. The model leverages the power of graph convolutions to effectively capture spatial relationships between body joints.

## Getting Started
To get started with the GraphEnet-v2 model, follow these steps:
1. **Install Dependencies**: Ensure you have Python and the required libraries installed. You can install the dependencies using pip:
   ```bash
   pip install -r requirements.txt
   ```
2. **How to install SplineConv**: To install the `torch-spline-conv` package, which is required for the GraphEnet-v2 model, you can use the following command:

```bash
# 1) Inspect your Torch and CUDA
python - <<'PY'
import torch, re
print("TORCH", torch.__version__.split('+')[0])
cuda = getattr(torch.version, "cuda", None)
tag = "cpu" if not cuda else "cu"+re.sub(r"\.","",cuda)[:3]
print("CUDA_TAG", tag)
PY
```
Replace `TORCH_X.Y.Z` and `CUDA_TAG` with the output of the above command
```bash
# 2) Install SplineConv
python -m pip install -U \
  torch-spline-conv \
  -f https://data.pyg.org/whl/torch-${TORCH_X.Y.Z}+${CUDA_TAG}.html

3. **How to install PyG binary wheels**: To install the PyTorch Geometric (PyG) binary wheels, you can use the following command:

```bash
# in the 'graphenet' venv
python -c "import torch, re; print(torch.__version__, getattr(torch.version,'cuda',None))"

# GPU build (you printed cu128 earlier)
pip install --no-build-isolation \
  torch-scatter torch-sparse torch-cluster \
  -f https://data.pyg.org/whl/torch-2.8.0+cu128.html
```
