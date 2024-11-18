def encode_rgb_to_bits(r, g, b):
    """
    Encodes RGB values into a 24-bit integer.

    Args:
        r (int): Red component (0–255).
        g (int): Green component (0–255).
        b (int): Blue component (0–255).

    Returns:
        int: A 24-bit integer representing the RGB value.
    """
    if not (0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255):
        raise ValueError("Each RGB component must be in the range 0–255.")

    return (r << 16) | (g << 8) | b


def decode_bits_to_rgb(encoded_rgb):
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


def encode_height_width_to_32bit(height, width):
    """
    Encodes two 16-bit integers into a single 32-bit integer.

    Args:
        height (int): The 16-bit integer for the high 16 bits (0–65535).
        width (int): The 16-bit integer for the low 16 bits (0–65535).

    Returns:
        int: A 32-bit integer combining the two 16-bit integers.
    """
    if not (0 <= height < 2 ** 16 and 0 <= width < 2 ** 16):
        raise ValueError("Both integers must be 16-bit (0–65535).")

    return (height << 16) | width


def decode_height_width_to_16bit(value):
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
