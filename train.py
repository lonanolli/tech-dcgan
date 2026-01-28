import argparse
import json
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms

from src.data import load_data, FlowerDataset
from src.models import Generator, Discriminator


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    img_dir = load_data(args.tgz_path, args.data_dir)

    transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    dataset = FlowerDataset(args.json_path, img_dir, transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    G = Generator(args.latent_dim).to(device)
    D = Discriminator().to(device)

    criterion = nn.BCELoss()
    optD = optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optG = optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))

    g_losses, d_losses = [], []

    for epoch in range(args.epochs):
        for images in loader:
            # Train D
            D.zero_grad(set_to_none=True)
            real = images.to(device)
            bsz = real.size(0)
            label_real = torch.full((bsz, 1), 1.0, device=device)
            label_fake = torch.full((bsz, 1), 0.0, device=device)

            out_real = D(real)
            errD_real = criterion(out_real, label_real)
            errD_real.backward()

            noise = torch.randn(bsz, args.latent_dim, 1, 1, device=device)
            fake = G(noise)
            out_fake = D(fake.detach())
            errD_fake = criterion(out_fake, label_fake)
            errD_fake.backward()
            optD.step()

            # Train G
            G.zero_grad(set_to_none=True)
            out_fake2 = D(fake)
            errG = criterion(out_fake2, label_real)
            errG.backward()
            optG.step()

        g_losses.append(errG.item())
        d_losses.append((errD_real + errD_fake).item())

        if args.log_every and (epoch + 1) % args.log_every == 0:
            print(
                f"Epoch {epoch+1}/{args.epochs} | Loss D: {d_losses[-1]:.4f} | Loss G: {g_losses[-1]:.4f}"
            )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save({"G": G.state_dict(), "D": D.state_dict()}, out_dir / "dcgan_weights.pt")
    with open(out_dir / "train_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    with open(out_dir / "losses.json", "w") as f:
        json.dump({"g_losses": g_losses, "d_losses": d_losses}, f, indent=2)


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--tgz-path", type=str, default="102flowers.tgz")
    p.add_argument("--json-path", type=str, default="category_to_images.json")
    p.add_argument("--data-dir", type=str, default="data/flower_data")
    p.add_argument("--out-dir", type=str, default="outputs")

    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--latent-dim", type=int, default=100)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=1)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    train(args)
