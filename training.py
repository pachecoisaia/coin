import torch
import tqdm
from collections import OrderedDict
from util import get_clamped_psnr
from encoding import encode_rgb_to_bits_tensor


class Trainer():
    def __init__(self, representation, lr=1e-3, print_freq=1):
        """
        Model to learn a representation of a single datapoint.

        Args:
            representation (siren.Siren): Neural net representation of image to be trained.
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

<<<<<<< Updated upstream
    def train(self, coordinates, features, num_iters):
        """Fit neural net to image.
=======
    def train(self, coordinates, features, encoded_features, num_iters):
        """
        Fit neural net to image.
>>>>>>> Stashed changes

        Args:
            coordinates (torch.Tensor): Tensor of coordinates. Shape (num_points, coordinate_dim).
            features (torch.Tensor): Tensor of RGB features. Shape (num_points, feature_dim).
            encoded_features (torch.Tensor): Tensor of encoded features as 32-bit integers. Shape (num_points, 1).
            num_iters (int): Number of iterations to train for.
        """
        with tqdm.trange(num_iters, ncols=100) as t:
            for i in t:
                # Update model
                self.optimizer.zero_grad()
                predicted = self.representation(coordinates)
<<<<<<< Updated upstream
                loss = self.loss_func(predicted, features)
                loss.backward()
                self.optimizer.step()

                # Calculate psnr
                psnr = get_clamped_psnr(predicted, features)

                # Print results and update logs
                log_dict = {'loss': loss.item(),
                            'psnr': psnr,
                            'best_psnr': self.best_vals['psnr']}
                t.set_postfix(**log_dict)
                for key in ['loss', 'psnr']:
                    self.logs[key].append(log_dict[key])

=======

                # Compute the loss on raw RGB features
                loss = self.loss_func(torch.clamp(predicted, 0, 1), features)  # Use raw features for gradients

                # Backpropagation
                loss.backward()
                self.optimizer.step()

                # Encode predicted RGB values for evaluation
                encoded_predicted = encode_rgb_to_bits_tensor((torch.clamp(predicted, 0, 1) * 255).to(torch.uint8))

                # Calculate PSNR using encoded values
                encoded_predicted_psnr = encoded_predicted.float()
                encoded_features_psnr = encoded_features.float()
                encoded_mse = torch.mean((encoded_predicted_psnr - encoded_features_psnr) ** 2)
                psnr = 20.0 * torch.log10(2**24 / torch.sqrt(encoded_mse))  # PSNR for 24-bit RGB

                # Print results and update logs
                log_dict = {'loss': loss.item(), 'psnr': psnr, 'best_psnr': self.best_vals['psnr']}
                t.set_postfix(**log_dict)
                for key in ['loss', 'psnr']:
                    self.logs[key].append(log_dict[key])

>>>>>>> Stashed changes
                # Update best values
                if loss.item() < self.best_vals['loss']:
                    self.best_vals['loss'] = loss.item()
                if psnr > self.best_vals['psnr']:
                    self.best_vals['psnr'] = psnr
<<<<<<< Updated upstream
                    # If model achieves best PSNR seen during training, update
                    # model
                    if i > int(num_iters / 2.):
                        for k, v in self.representation.state_dict().items():
                            self.best_model[k].copy_(v)
=======
                    # If model achieves best PSNR seen during training, update model
                    if i > int(num_iters / 2.):
                        for k, v in self.representation.state_dict().items():
                            self.best_model[k].copy_(v)
>>>>>>> Stashed changes
