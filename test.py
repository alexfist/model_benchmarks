import torch
import torch_scatter
import torch_geometric
import torch_sparse
import torch_cluster
from minimol import Minimol
print("torch version:", torch.__version__)
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu:', torch.cuda.get_device_name(0))
print('All imports successful!')