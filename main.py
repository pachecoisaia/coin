import argparse
import imageio.v2 as imageio
import os
import random
import torch
import util
from siren import Siren
from torchvision import transforms
from torchvision.utils import save_image
from training import Trainer
from encoding import encode_rgb_to_bits_tensor, decode_bits_to_rgb  # Import encoding functions
import json

def tensor_to_serializable(tensor):
    """
    Converts a tensor into a JSON-serializable format.
    """
    if tensor.numel() == 1:  # Single-element tensor
        return tensor.item()
    else:  # Multi-element tensor
        return tensor.tolist()

parser = argparse.ArgumentParser()
parser.add_argument("-ld", "--logdir", help="Path to save logs", default=r"C:\Users\ziyad\OneDrive\Desktop\output_image")
parser.add_argument("-ni", "--num_iters", help="Number of iterations to train for", type=int, default=500)
parser.add_argument("-lr", "--learning_rate", help="Learning rate", type=float, default=2e-4)
parser.add_argument("-se", "--seed", help="Random seed", type=int, default=random.randint(1, int(1e6)))
parser.add_argument("-fd", "--full_dataset", help="Whether to use full dataset", action='store_true')
parser.add_argument("-iid", "--image_id", help="Image ID to train on, if not the full dataset", type=int, default=12)
parser.add_argument("-lss", "--layer_size", help="Layer sizes as list of ints", type=int, default=28)
parser.add_argument("-nl", "--num_layers", help="Number of layers", type=int, default=10)
parser.add_argument("-w0", "--w0", help="w0 parameter for SIREN model.", type=float, default=30.0)
parser.add_argument("-w0i", "--w0_initial", help="w0 parameter for first layer of SIREN model.", type=float, default=30.0)

args = parser.parse_args()

# Set up torch and cuda
dtype = torch.float32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_tensor_type('torch.cuda.FloatTensor' if torch.cuda.is_available() else 'torch.FloatTensor')

# Set random seeds
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

# Determine dataset range
if args.full_dataset:
    min_id, max_id = 1, 24  # Kodak dataset runs from kodim01.png to kodim24.png
else:
    min_id, max_id = args.image_id, args.image_id

# Dictionary to register metrics
results = {'fp_bpp': [], 'hp_bpp': [], 'fp_psnr': [], 'hp_psnr': []}

# Create directory to store experiments
if not os.path.exists(args.logdir):
    os.makedirs(args.logdir)

# Process each image
for i in range(min_id, max_id + 1):
    print(f"Processing image {i}...")

    # Load image
    img = imageio.imread(f"kodak-dataset/kodim{str(i).zfill(2)}.png")
    img = transforms.ToTensor()(img).float().to(device, dtype)

    # Encode ground truth RGB to 32-bit integers (optional)
    encoded_img = encode_rgb_to_bits_tensor((img.permute(1, 2, 0) * 255).to(torch.uint8).reshape(-1, 3))

    # Setup model
    func_rep = Siren(
        dim_in=2,
        dim_hidden=args.layer_size,
        dim_out=3,
        num_layers=args.num_layers,
        final_activation=torch.nn.Identity(),
        w0_initial=args.w0_initial,
        w0=args.w0
    ).to(device)

    # Set up training
    trainer = Trainer(func_rep, lr=args.learning_rate)
    coordinates, features = util.to_coordinates_and_features(img)
    coordinates, features = coordinates.to(device, dtype), features.to(device, dtype)

    # Encode RGB features into 32-bit packed integers
    encoded_features = encode_rgb_to_bits_tensor((features * 255).to(torch.uint8))

    # Train model in full precision
    trainer.train(coordinates, features, num_iters=args.num_iters)
    print(f'Best training psnr: {trainer.best_vals["psnr"]:.2f}')

    # Log full precision results
    results['fp_bpp'].append(fp_bpp)
    results['fp_psnr'].append(trainer.best_vals['psnr'])

    # Save best model
    torch.save(trainer.best_model, os.path.join(args.logdir, f"best_model_{i}.pt"))

    # Save full-precision reconstructed image
    with torch.no_grad():
        img_recon = func_rep(coordinates).reshape(img.shape[1], img.shape[2], 3).permute(2, 0, 1)
        save_image(torch.clamp(img_recon, 0, 1).to("cpu"), os.path.join(args.logdir, f"fp_reconstruction_{i}.png"))

        # Encode reconstructed RGB into 32-bit integers
        encoded_recon = encode_rgb_to_bits_tensor((img_recon.permute(1, 2, 0) * 255).to(torch.uint8).reshape(-1, 3))

    # Half-precision support (only if CUDA is available)
    if torch.cuda.is_available():
        func_rep = func_rep.half().to("cuda")
        coordinates = coordinates.half().to("cuda")

        # Calculate model size (HP)
        hp_bpp = util.bpp(model=func_rep, image=img)
        results['hp_bpp'].append(hp_bpp)
        print(f"Half precision bpp: {hp_bpp:.2f}")

        # Compute half-precision reconstruction and PSNR
        with torch.no_grad():
            img_recon_hp = func_rep(coordinates).reshape(img.shape[1], img.shape[2], 3).permute(2, 0, 1).float()
            hp_psnr = util.get_clamped_psnr(img_recon_hp, img)
            save_image(torch.clamp(img_recon_hp, 0, 1).to("cpu"), os.path.join(args.logdir, f"hp_reconstruction_{i}.png"))
            print(f"Half precision PSNR: {hp_psnr:.2f}")
            results['hp_psnr'].append(hp_psnr)
    else:
        results['hp_bpp'].append(fp_bpp)
        results['hp_psnr'].append(0.0)

    # Save per-image logs
    serializable_logs = {key: [tensor_to_serializable(v) for v in value] for key, value in trainer.logs.items()}
    with open(os.path.join(args.logdir, f"logs_{i}.json"), "w") as f:
        json.dump(serializable_logs, f)

    print("\n")

# Save overall results
print("Full results:")
print(results)
with open(os.path.join(args.logdir, "results.json"), "w") as f:
    json.dump(results, f)

# Save aggregated results
results_mean = {key: util.mean(results[key]) for key in results}
with open(os.path.join(args.logdir, "results_mean.json"), "w") as f:
    json.dump(results_mean, f)

print("Aggregate results:")
print(f"Full precision: bpp = {results_mean['fp_bpp']:.2f}, PSNR = {results_mean['fp_psnr']:.2f}")
print(f"Half precision: bpp = {results_mean['hp_bpp']:.2f}, PSNR = {results_mean['hp_psnr']:.2f}")