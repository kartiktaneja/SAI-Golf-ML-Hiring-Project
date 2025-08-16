import torch

print("Number of GPUs:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")


print("Current cuda device: ", torch.cuda.current_device())
print("Cuda device name: ", torch.cuda.get_device_name(torch.cuda.current_device()))