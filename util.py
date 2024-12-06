import numpy as np
import torch
from torch._C import dtype
from typing import Dict

from encoding import encode_rgb_to_bits_tensor, encode_height_width_to_32bit_tensor

DTYPE_BIT_SIZE: Dict[dtype, int] = {
    torch.float32: 32,
    torch.float: 32,
    torch.float64: 64,
    torch.double: 64,
    torch.float16: 16,
    torch.half: 16,
    torch.bfloat16: 16,
    torch.complex32: 32,
    torch.complex64: 64,
    torch.complex128: 128,
    torch.cdouble: 128,
    torch.uint8: 8,
    torch.int8: 8,
    torch.int16: 16,
    torch.short: 16,
    torch.int32: 32,
    torch.int: 32,
    torch.int64: 64,
    torch.long: 64,
    torch.bool: 1
}


def to_coordinates_and_features(img):
    """Converts an image to a set of coordinates and features.

    Args:
        img (torch.Tensor): Shape (channels, height, width).
    """
    # Coordinates are indices of all non zero locations of a tensor of ones of
    # same shape as spatial dimensions of image
    height, width = img.shape[1], img.shape[2]

    # Generate the x (width) and y (height) coordinates
    x_coords = torch.arange(width, dtype=torch.int32).repeat(height, 1)  # (height, width)
    y_coords = torch.arange(height, dtype=torch.int32).repeat(width, 1).T  # (height, width)

    coordinates = encode_height_width_to_32bit_tensor(x_coords, y_coords).flatten().unsqueeze(1).flatten()

    # Convert float RGB Features to unsigned integers
    features = (img * 255).to(dtype=torch.uint8).permute(1, 2, 0)

    # Encode Features
    features = encode_rgb_to_bits_tensor(features).flatten()

    return coordinates, features


def model_size_in_bits(model):
    """Calculate total number of bits to store `model` parameters and buffers."""
    return sum(sum(t.nelement() * DTYPE_BIT_SIZE[t.dtype] for t in tensors)
               for tensors in (model.parameters(), model.buffers()))


def bpp(image, model):
    """Computes size in bits per pixel of model.

    Args:
        image (torch.Tensor): Image to be fitted by model.
        model (torch.nn.Module): Model used to fit image.
    """
    num_pixels = np.prod(image.shape) / 3  # Dividing by 3 because of RGB channels
    return model_size_in_bits(model=model) / num_pixels


def psnr(img1, img2):
    """Calculates PSNR between two images.

    Args:
        img1 (torch.Tensor):
        img2 (torch.Tensor):
    """
    return 20. * np.log10(1.) - 10. * (img1 - img2).detach().pow(2).mean().log10().to('cpu').item()


def clamp_image(img):
    """Clamp image values to like in [0, 1] and convert to unsigned int.

    Args:
        img (torch.Tensor):
    """
    # Values may lie outside [0, 1], so clamp input
    img_ = torch.clamp(img, 0., 1.)
    # Pixel values lie in {0, ..., 255}, so round float tensor
    return torch.round(img_ * 255) / 255.


def get_clamped_psnr(img, img_recon):
    """Get PSNR between true image and reconstructed image. As reconstructed
    image comes from output of neural net, ensure that values like in [0, 1] and
    are unsigned ints.

    Args:
        img (torch.Tensor): Ground truth image.
        img_recon (torch.Tensor): Image reconstructed by model.
    """
    return psnr(img, clamp_image(img_recon))


def mean(list_):
    return np.mean(list_)


def remove_msb_tensor(tensor):
    # Compute the bit length for each element in the tensor
    if tensor.dtype not in [torch.int16, torch.int32 or torch.int64]:
        raise ValueError("tensor must be signed integer 32-bit/64-bit tensors.")

    numBits = 0

    if tensor.dtype == torch.int16:
        numBits = 16

    if tensor.dtype == torch.int32:
        numBits = 32

    if tensor.dtype == torch.int64:
        numBits = 64

    n = numBits - 24

    # Generate a mask to keep the lower (bit_length - n) bits
    mask = (1 << (numBits - n)) - 1

    # Apply the mask to the tensor and return the result
    return tensor & mask
