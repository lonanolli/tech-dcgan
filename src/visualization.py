import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
from torchvision import utils


def plot_losses(g_losses, d_losses):
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(g_losses, label="Generator (G)", color="blue", alpha=0.7)
    plt.plot(d_losses, label="Discriminator (D)", color="orange", alpha=0.7)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()


@torch.no_grad()
def sample_images_grid(G, latent_dim: int, device, n: int = 10, nrow: int = 10):
    G.eval()
    noise = torch.randn(n, latent_dim, 1, 1, device=device)
    fake = G(noise).detach().cpu()
    fake = (fake + 1) / 2
    grid = utils.make_grid(fake, nrow=nrow)
    plt.figure(figsize=(15, 3))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")
    plt.tight_layout()
    return noise.squeeze(-1).squeeze(-1).cpu().numpy()
