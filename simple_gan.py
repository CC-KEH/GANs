import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter


class Discriminator(nn.Module):
    def __init__(self, img_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(img_dim, 120),
            nn.LeakyReLU(0.1),
            nn.Linear(120, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(img_dim, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, img_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


# Hyperparameters
device = 'cuda' if torch.cuda.is_available() else 'cpu'
lr = 3e-4
z_dim = 64
img_dim = 28 * 28 * 1
batch_size = 32
num_epochs = 100


disc = Discriminator(img_dim=img_dim).to(device)
gen = Generator(z_dim=z_dim, img_dim=img_dim).to(device)
fixed_noise = torch.randn(batch_size, z_dim).to(device)
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)
dataset = datasets.MNIST(root='dataset/', transforms=transform, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
opt_disc = optim.Adam(disc.parameters(), lr=lr)
opt_gen = optim.Adam(gen.parameters(), lr=lr)
criterion = nn.BCELoss()
writer_fake = SummaryWriter(f'runs/GAN_MNIST/fake')
writer_real = SummaryWriter(f'runs/GAN_MNIST/real')
step = 0


for epoch in range(num_epochs):
    for batch_idx, (img, label) in enumerate(loader):
        img = img.view(-1, 784).to(device)
        batch_size = img.shape[0]

        # * Train Discriminator
        # * max log(D(real)) + log(1-D(G(z)))

        noise = torch.randn(batch_size, z_dim).to(device)
        fake = gen(noise)

        # log(D(real)) + 0
        disc_real = disc(img).view(-1)
        lossD_real = criterion(disc_real, torch.ones_like(disc_real))

        # 0 + log(1-D(G(z)))
        disc_fake = disc(fake).view(-1)
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))

        loss_D = (lossD_real + lossD_fake)/2
        disc.zero_grad()
        loss_D.backward(retain_graph=True)
        opt_disc.step()

        # * Train Generator
        # * min log(1-D(G(z))) <-> max log(D(G(z)))

        output = disc(fake).view(-1)
        loss_G = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        loss_G.backward()
        opt_gen.step()

        if batch_idx == 0:
            print(
                f"Epoch: [{epoch}/{num_epochs}] \ "
                f"Loss D: {loss_D:.4f}, Loss G: {loss_G:.4f}"
            )
            with torch.no_grad():
                fake = gen(fixed_noise).reshape(-1, 1, 28, 28)
                data = img.reshape(-1, 1, 28, 28)
                img_grid_fake = torchvision.utils.make_grid(
                    fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(
                    data, normalize=True)
                writer_fake.add_image(
                    "MNIST Fake Images", img_grid_fake, global_step=step
                )
                writer_real.add_image(
                    "MNIST Real Images", img_grid_real, global_step=step
                )
                step += 1
