import torch
import torch.cuda


def encode_rgb_to_bits_tensor(rgb_tensor):
    # Extract R, G, B channels
    r = rgb_tensor[:, :, 0].to(torch.int32)
    g = rgb_tensor[:, :, 1].to(torch.int32)
    b = rgb_tensor[:, :, 2].to(torch.int32)

    # Perform bitwise encoding
    value = (r << 16) | (g << 8) | b

    return value


def decode_bits_to_rgb(encoded_tensor):

    r_mask = 0xFF0000
    g_mask = 0xFF00
    b_mask = 0xFF

    # Decode R, G, B channels
    r = (encoded_tensor & r_mask) >> 16
    g = (encoded_tensor & g_mask) >> 8
    b = (encoded_tensor) & b_mask

    # Stack the decoded values into an RGB tensor
    decoded_rgb = torch.stack((r, g, b), dim=1)

    return decoded_rgb


def encode_height_width_to_32bit_tensor(width, height):
    """
    Encodes two tensors of 32-bit integers into a single tensor of 32-bit floats after normalization.

    Args:
        heights (Tensor): A tensor of height values (16-bit integers).
        widths (Tensor): A tensor of width values (16-bit integers).

    Returns:
        Tensor: A tensor of normalized 32-bit floats.
    """
    # Ensure heights and widths are tensors of integer type

    if width.dtype not in [torch.int32, torch.int64] or height.dtype not in [torch.int16, torch.int32, torch.int64]:
        raise ValueError("Heights and widths must be signed integer 16 bit tensors.")

    max_bits = {
        torch.int32: 16,
        torch.int64: 32
    }
    num_bits = max_bits[width.dtype]

    # Combine width and height into a single integer tensor
    combined = (width << num_bits) | height


    return combined


def decode_height_width_to_16bit(value):
    # TODO Make Tensor Friendly
    """
    Decodes a 32-bit integer into two 16-bit integers.

    Args:
        value (int): The 32-bit integer to decode.

    Returns:
        tuple: A tuple of two integers (height, width), each 16 bits.
    """
    if not (0 <= value < 2 ** 32):
        raise ValueError("Value must be a 32-bit integer (0–4,294,967,295).")

    height = (value >> 16) & 0xFFFF  # Extract the high 16 bits (height)
    width = value & 0xFFFF  # Extract the low 16 bits (width)
    return height, width