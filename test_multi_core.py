import torch
from multiprocessing import Pool
import os

def test_torch_task(x):
    print(f"Process {os.getpid()} handling task {x}")
    tensor = torch.zeros((10**4, 10**4))
    result = tensor.sum().item()
    return result

if __name__ == "__main__":
    num_cores = 8
    with Pool(processes=num_cores) as pool:
        results = pool.map(test_torch_task, range(num_cores * 10000))
    print("Results:", results)