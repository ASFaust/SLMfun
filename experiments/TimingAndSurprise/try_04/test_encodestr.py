import torch
import numpy as np
import time

# Original function
def encode_string_original(to_encode, device, truncate=True):
    to_encode_len = len(to_encode)
    encoded = list(to_encode.encode('utf-8'))
    encoded = np.eye(256)[encoded]

    if truncate:
        encoded = encoded[-to_encode_len:]

    encoded = torch.from_numpy(encoded).float().to(device)
    return encoded

# Optimized function
def encode_string_optimized(to_encode, device, truncate=True):
    encoded_bytes = torch.tensor(list(to_encode.encode('utf-8')), dtype=torch.long, device=device)
    one_hot_encoded = torch.zeros((encoded_bytes.size(0), 256), dtype=torch.float32, device=device)
    one_hot_encoded.scatter_(1, encoded_bytes.unsqueeze(1), 1.0)

    if truncate:
        one_hot_encoded = one_hot_encoded[-len(to_encode):]

    return one_hot_encoded

# Testing and benchmarking
def benchmark_encode_string():
    test_string = "Hello, world!" * 1000  # A long test string
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Measure original function
    start_time = time.time()
    encode_string_original(test_string, device)
    original_time = time.time() - start_time

    # Measure optimized function
    start_time = time.time()
    encode_string_optimized(test_string, device)
    optimized_time = time.time() - start_time

    print(f"Original function time: {original_time:.6f} seconds")
    print(f"Optimized function time: {optimized_time:.6f} seconds")

benchmark_encode_string()
