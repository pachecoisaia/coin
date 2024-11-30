import torch
import tqdm
from collections import OrderedDict
from util import get_clamped_psnr


class Trainer():
    def __init__(self, representation, lr=1e-3, print_freq=1):
        """Model to learn a representation of a single datapoint.

        Args:
            representation (siren.Siren): Neural net representation of image to
                be trained.
            lr (float): Learning rate to be used in Adam optimizer.
            print_freq (int): Frequency with which to print losses.
        """
        self.representation = representation
        self.optimizer = torch.optim.Adam(self.representation.parameters(), lr=lr)
        self.print_freq = print_freq
        self.steps = 0  # Number of steps taken in training
        self.loss_func = torch.nn.MSELoss()
        self.best_vals = {'psnr': 0.0, 'loss': 1e8}
        self.logs = {'psnr': [], 'loss': []}
        # Store parameters of best model (in terms of highest PSNR achieved)
        self.best_model = OrderedDict((k, v.detach().clone()) for k, v in self.representation.state_dict().items())

    def train(self, coordinates, features, num_iters):
        """Fit neural net to image.

        Args:
            coordinates (torch.Tensor): Tensor of coordinates.
                Shape (num_points, coordinate_dim).
            features (torch.Tensor): Tensor of features. Shape (num_points, feature_dim).
            num_iters (int): Number of iterations to train for.
        """

        batch_size = 32  # Set an appropriate batch size based on your memory capacity
        dataset = torch.utils.data.TensorDataset(coordinates, features)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                                                  pin_memory=False)

        with tqdm.trange(num_iters, ncols=100) as t:
            for i in t:
                # Loop through batches
                for batch_coords, batch_features in data_loader:
                    # Move data to GPU
                    batch_coords, batch_features = batch_coords.to('cuda'), batch_features.to('cuda')

                    # Update model
                    self.optimizer.zero_grad()
                    predicted = self.representation(batch_coords)
                    loss = self.loss_func(predicted, batch_features)
                    loss.backward()
                    self.optimizer.step()

                    # Calculate PSNR for current batch
                    psnr = get_clamped_psnr(predicted, batch_features)

                    # Print results and update logs
                    log_dict = {'loss': loss.item(),
                                'psnr': psnr,
                                'best_psnr': self.best_vals['psnr']}
                    t.set_postfix(**log_dict)
                    for key in ['loss', 'psnr']:
                        self.logs[key].append(log_dict[key])

                    # Update best values
                    if loss.item() < self.best_vals['loss']:
                        self.best_vals['loss'] = loss.item()
                    if psnr > self.best_vals['psnr']:
                        self.best_vals['psnr'] = psnr
                        # If model achieves best PSNR seen during training, update
                        # model
                        if i > int(num_iters / 2.):
                            for k, v in self.representation.state_dict().items():
                                self.best_model[k].copy_(v)

                # Update the progress bar for every iteration (i.e., per batch)
                t.set_description(f'Epoch {i}/{num_iters}, Loss: {log_dict["loss"]:.4f}, PSNR: {log_dict["psnr"]:.4f}')
