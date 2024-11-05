import numpy as np
import torch

def index_to_onehot(index):
    onehot = torch.zeros(256)
    onehot[index] = 1
    return onehot


def sample_from_logits(logits, temperature=1.0, top_k=8):
    """
    Samples a one-hot vector from a tensor of logits with top-k sampling strategy.

    :param logits: a tensor of shape [1, 256]
    :param temperature: scaling factor to adjust probabilities
    :param top_k: the number of top logits to select from
    :return: a one-hot vector corresponding to the sampled index
    """
    if temperature == 0:
        return index_to_onehot(torch.argmax(logits[0]))

    logits = logits.reshape(1, -1)  # reshaping to [1, num_classes], where num_classes=256 in your case
    logits = logits / temperature

    # Extract the top-k logits and their corresponding indices
    top_k_logits, top_k_indices = torch.topk(logits, k=top_k, dim=-1)

    # Sample from the top-k logits
    top_k_probabilities = torch.softmax(top_k_logits, dim=-1)
    sampled_index_in_top_k = torch.multinomial(top_k_probabilities, 1)[0, 0]

    # Convert the sampled index back to the original range of indices
    sampled_index = top_k_indices[0, sampled_index_in_top_k]
    return index_to_onehot(sampled_index)

def encode_string(to_encode, device, truncate = True):
    """
    Encodes a unicode string of l bytes into a one-hot tensor of shape [l, 256]
    :param to_encode:
    :param device:
    :param truncate: If true, the returned tensor will be truncated to the length of the string.
    Some unicode characters take more than one byte, so the length of the string is not necessarily the same as the number of bytes.
    :return:
    """
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
    """
    Decodes a one-hot tensor of shape [l, 256] into a unicode string of l bytes.
    :param encoded_tensor:
    :return:
    """
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
