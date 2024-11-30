import torch


def encode_rgb_to_bits_tensor(rgb_tensor):
    """
    Encodes RGB values into a 24-bit integer by shifting and combining the channels
    for the entire tensor (vectorized approach).

    Args:
        rgb_tensor (Tensor): A tensor of shape (N, 3) containing RGB values (r, g, b).

    Returns:
        Tensor: A tensor of encoded 24-bit integers.
    """
    # Split the tensor into R, G, B components
    r, g, b = rgb_tensor[:, 0], rgb_tensor[:, 1], rgb_tensor[:, 2]

    if r.dtype not in [torch.uint8] or g.dtype not in [torch.uint8] or b.dtype not in [torch.uint8]:
        raise ValueError("RGB must be unsigned integer 8 bit tensors.")

    # Perform the encoding (bitwise shift and OR) for the entire tensor
    encoded = (r << 16) | (g << 8) | b

    return encoded


def decode_bits_to_rgb(encoded_rgb):
    # TODO Make Tensor Friendly
    """
    Decodes a 24-bit integer into its RGB components.

    Args:
        encoded_rgb (int): A 24-bit integer representing an RGB value.

    Returns:
        tuple: A tuple of three integers (r, g, b), each in the range 0–255.
    """
    if not (0 <= encoded_rgb <= 0xFFFFFF):
        raise ValueError("Encoded RGB value must be a 24-bit integer (0–16777215).")

    # Extract each color component
    r = (encoded_rgb >> 16) & 0xFF  # Red is the highest 8 bits
    g = (encoded_rgb >> 8) & 0xFF  # Green is the middle 8 bits
    b = encoded_rgb & 0xFF  # Blue is the lowest 8 bits

    return r, g, b


def encode_normalize_height_width_to_32bit_tensor(heights, widths):
    """
    Encodes two tensors of 16-bit integers into a single tensor of 32-bit floats after normalization.

    Args:
        heights (Tensor): A tensor of height values (16-bit integers).
        widths (Tensor): A tensor of width values (16-bit integers).

    Returns:
        Tensor: A tensor of normalized 32-bit floats.
    """
    # Ensure heights and widths are tensors of integer type

    if heights.dtype not in [torch.int16] or widths.dtype not in [torch.int16]:
        raise ValueError("Heights and widths must be signed integer 16 bit tensors.")

    # Perform bit-shifting and OR operation to combine into 32-bit integer (vectorized)
    combined = (heights << 16) | widths  # Perform bitwise shift and OR

    # Normalize the resulting 32-bit value
    max_32bit_unsigned = 2 ** 32 - 1  # Maximum possible value for an unsigned 32-bit integer
    normalized = combined.float() / max_32bit_unsigned  # Convert to float and normalize

    return normalized


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
