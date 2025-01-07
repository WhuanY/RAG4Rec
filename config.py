import torch

print(torch.__version__)  # Check the PyTorch version
print(torch.version.cuda)  # Check the CUDA version it was built with
print(torch.cuda.is_available())  # Should return True
