import numpy as np
import torch


def encode_string(to_encode, device, truncate = True):
    to_encode_len = len(to_encode)
    # Convert string to bytes
    encoded = list(to_encode.encode('utf-8'))

    # Convert each byte to a one-hot vector
    encoded = np.eye(256)[encoded]

    if truncate:
        encoded = encoded[-to_encode_len:]

    # Convert to a PyTorch tensor
    encoded = torch.from_numpy(encoded).float().to(device)

    #return shape is [to_encode_len, 256]
    return encoded


def decode_string(encoded_tensor):
    # Convert tensor back to numpy
    decoded_np = encoded_tensor.cpu().numpy()

    # Find original byte values
    byte_values = np.argmax(decoded_np, axis=1).tolist()

    # Convert byte values to a byte array.
    byte_arr = bytearray(byte_values)

    # Convert byte array to UTF-8 string. ignore errors because some byte values are invalid
    #decoded_str = byte_arr.decode('utf-8')

    decoded_str = byte_arr.decode('utf-8', errors='ignore')

    return decoded_str
