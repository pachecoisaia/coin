import numpy as np
import torch
from typing import Dict

<<<<<<< Updated upstream

DTYPE_BIT_SIZE: Dict[dtype, int] = {
=======
DTYPE_BIT_SIZE: Dict[torch.dtype, int] = {
>>>>>>> Stashed changes
    torch.float32: 32,
    torch.float: 32,
    torch.float64: 64,
    torch.double: 64,
    torch.float16: 16,
    torch.half: 16,
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
    """Converts a 3-channel image to a set of coordinates and features.

    Args:
        img (torch.Tensor): Shape (3, height, width) representing an RGB image.
    """
<<<<<<< Updated upstream
    # Coordinates are indices of all non zero locations of a tensor of ones of
    # same shape as spatial dimensions of image
    coordinates = torch.ones(img.shape[1:]).nonzero(as_tuple=False).float()
    # Normalize coordinates to lie in [-.5, .5]
    coordinates = coordinates / (img.shape[1] - 1) - 0.5
    # Convert to range [-1, 1]
    coordinates *= 2
    # Convert image to a tensor of features of shape (num_points, channels)
    features = img.reshape(img.shape[0], -1).T
=======
    height, width = img.shape[1], img.shape[2]

    # Generate grid of normalized coordinates
    y = torch.linspace(-1, 1, steps=height)
    x = torch.linspace(-1, 1, steps=width)
    grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
    coordinates = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)

    # Flatten the RGB image into features
    features = img.permute(1, 2, 0).reshape(-1, 3)  # Shape: (height * width, 3)
>>>>>>> Stashed changes
    return coordinates, features


def bpp(image, model):
    """Computes size in bits per pixel of the model.

    Args:
        image (torch.Tensor): 3-channel image to be fitted by the model.
        model (torch.nn.Module): Model used to fit the image.
    """
    num_pixels = image.shape[1] * image.shape[2]  # Total spatial pixels
    return model_size_in_bits(model=model) / num_pixels


def rgb_to_32bit(img):
    """Encodes a 3-channel RGB image into a 32-bit packed format."""
    r, g, b = img[0], img[1], img[2]
    packed = (r * 255).long() << 16 | (g * 255).long() << 8 | (b * 255).long()
    return packed.float()


def unpack_32bit_to_rgb(packed_img):
    """Decodes a 32-bit packed image back into 3 RGB channels."""
    r = (packed_img >> 16) & 255
    g = (packed_img >> 8) & 255
    b = packed_img & 255
    return torch.stack([r, g, b], dim=0) / 255.0  # Normalize to [0, 1]


def psnr(img1, img2):
    """Calculates PSNR between two images."""
    return 20. * np.log10(1.) - 10. * (img1 - img2).detach().pow(2).mean().log10().to('cpu').item()


def clamp_image(img):
    """Clamp image values to lie in [0, 1]."""
    return torch.clamp(img, 0., 1.)


def get_clamped_psnr(img, img_recon):
    """Get PSNR between true image and reconstructed image."""
    return psnr(img, clamp_image(img_recon))


def model_size_in_bits(model):
    """Calculate total number of bits to store model parameters and buffers."""
    return sum(
        sum(t.nelement() * DTYPE_BIT_SIZE[t.dtype] for t in tensors)
        for tensors in (model.parameters(), model.buffers())
    )


def mean(list_):
<<<<<<< Updated upstream
    return np.mean(list_)
=======
    """Compute mean of a list."""
    return np.mean(list_)
>>>>>>> Stashed changes
