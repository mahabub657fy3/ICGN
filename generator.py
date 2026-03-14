import torch
import torch.nn as nn


ngf = 64

class lowpass(nn.Module):
    def __init__(self, channels=3, k=None, kernel_size=7):
        super().__init__()

        # Robust handling for k=None or k<=0
        if k is not None and k > 0:
            kernel_size = 4 * k + 1
            sigma = float(k)
        else:
            sigma = 2.0  # reasonable default

        self.kernel_size = kernel_size
        self.channels = channels

        x = torch.arange(kernel_size).float() - kernel_size // 2
        gauss = torch.exp(-(x ** 2) / (2 * sigma ** 2))
        gauss = gauss / gauss.sum()
        kernel_2d = gauss[:, None] * gauss[None, :]
        kernel_2d = kernel_2d / kernel_2d.sum()

        weight = kernel_2d[None, None, :, :].repeat(channels, 1, 1, 1)

        self.pad = nn.ReflectionPad2d(kernel_size // 2)
        self.gaussian_filter = nn.Conv2d(channels, channels, kernel_size, stride=1, padding=0, groups=channels, bias=False)
        self.gaussian_filter.weight.data.copy_(weight)
        self.gaussian_filter.weight.requires_grad_(False)

    def forward(self, x):
        return self.gaussian_filter(self.pad(x))


class MDM(nn.Module):
    def __init__(self, cond_dim: int, num_features: int):
        super().__init__()
        self.num_features = num_features
        self.fc = nn.Linear(cond_dim, 2 * num_features)

        # Initialize to identity modulation (gamma=0, beta=0)
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gb = self.fc(cond)  # [B, 2C]
        gamma, beta = gb.chunk(2, dim=1)
        gamma = gamma.view(-1, self.num_features, 1, 1)
        beta = beta.view(-1, self.num_features, 1, 1)
        return (1.0 + gamma) * x + beta


class MDMResnetBlock(nn.Module):
    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.pad1 = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=False)
        self.in1 = nn.InstanceNorm2d(dim, affine=False)
        self.film1 = MDM(cond_dim, dim)
        self.relu = nn.ReLU(inplace=True)

        self.pad2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=False)
        self.in2 = nn.InstanceNorm2d(dim, affine=False)
        self.film2 = MDM(cond_dim, dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.pad1(x))
        h = self.in1(h)
        h = self.film1(h, cond)
        h = self.relu(h)

        h = self.conv2(self.pad2(h))
        h = self.in2(h)
        h = self.film2(h, cond)

        return x + h


class MDMGenerator(nn.Module):
    def __init__(self, inception: bool = False, nz: int = 16, k=None, device=None,
                 cond_dim: int = 512, n_resblocks: int = 6):
        super().__init__()
        self.inception = inception
        self.device = device
        self.nz = nz

        self.lowpass = lowpass(k=k).to(device) if device is not None else lowpass(k=k)

        # Encoder
        self.pad1 = nn.ReflectionPad2d(3)
        self.conv1 = nn.Conv2d(3, ngf, kernel_size=7, stride=1, padding=0, bias=False)
        self.in1 = nn.InstanceNorm2d(ngf, affine=False)
        self.film1 = MDM(cond_dim, ngf)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.in2 = nn.InstanceNorm2d(ngf * 2, affine=False)
        self.film2 = MDM(cond_dim, ngf * 2)

        self.conv3 = nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1, bias=False)
        self.in3 = nn.InstanceNorm2d(ngf * 4, affine=False)
        self.film3 = MDM(cond_dim, ngf * 4)

        # Bottleneck
        self.resblocks = nn.ModuleList([MDMResnetBlock(ngf * 4, cond_dim) for _ in range(n_resblocks)])

        # Decoder
        self.deconv1 = nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.in4 = nn.InstanceNorm2d(ngf * 2, affine=False)
        self.film4 = MDM(cond_dim, ngf * 2)

        self.deconv2 = nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.in5 = nn.InstanceNorm2d(ngf, affine=False)
        self.film5 = MDM(cond_dim, ngf)

        self.pad_out = nn.ReflectionPad2d(3)
        self.conv_out = nn.Conv2d(ngf, 3, kernel_size=7, stride=1, padding=0)
        self.tanh = nn.Tanh()

    def forward(self, input: torch.Tensor, cond: torch.Tensor, eps: float = 16 / 255.0) -> torch.Tensor:
        """
        input: [B,3,H,W] in [0,1]
        cond:  [B,512] (CLIP text feature from conditioner)
        """
        B, C, H, W = input.shape

        # Low-frequency base
        input_low = self.lowpass(input)

        # Encoder
        x = self.conv1(self.pad1(input))
        x = self.in1(x)
        x = self.film1(x, cond)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.in2(x)
        x = self.film2(x, cond)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.in3(x)
        x = self.film3(x, cond)
        x = self.relu(x)

        # Bottleneck
        for blk in self.resblocks:
            x = blk(x, cond)

        # Decoder
        x = self.deconv1(x)
        x = self.in4(x)
        x = self.film4(x, cond)
        x = self.relu(x)

        x = self.deconv2(x)
        x = self.in5(x)
        x = self.film5(x, cond)
        x = self.relu(x)

        x = self.conv_out(self.pad_out(x))
        x = self.tanh(x)  # [-1,1]

        # Crop to match input size (important for odd sizes like 255)
        if x.shape[2] != H or x.shape[3] != W:
            x = x[:, :, :H, :W]

        # Scale to eps and constrain in L∞ ball
        delta = x * eps
        adv = input_low + delta
        adv = torch.min(torch.max(adv, input - eps), input + eps)

        return adv
