import torch
import torch.nn as nn


class autoencoder(nn.Module):
    def __init__(self, h_dim):
        super(autoencoder, self).__init__()
        self.z_dim = 28*28 #128
        self.h_dim = h_dim
        self.h2_dim = 64
        self.encoder = nn.Sequential(
            nn.Linear(self.z_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.ReLU(True), nn.Linear(128, self.h2_dim), nn.ReLU(True), nn.Linear(self.h2_dim, self.h_dim), nn.BatchNorm1d(self.h_dim, affine=False))
        self.decoder = nn.Sequential(
            nn.Linear(self.h_dim, self.h2_dim),
            nn.ReLU(True),
            nn.Linear(self.h2_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True), nn.Linear(256, 28 * 28), nn.Tanh())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class autoencoder_nobatchnorm(nn.Module):
    def __init__(self, h_dim):
        super(autoencoder_nobatchnorm, self).__init__()
        self.z_dim = 28*28 #128
        self.h_dim = h_dim
        self.h2_dim = 64
        self.encoder = nn.Sequential(
            nn.Linear(self.z_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.ReLU(True), nn.Linear(128, self.h2_dim), nn.ReLU(True), nn.Linear(self.h2_dim, self.h_dim))
        self.decoder = nn.Sequential(
            nn.Linear(self.h_dim, self.h2_dim),
            nn.ReLU(True),
            nn.Linear(self.h2_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True), nn.Linear(256, 28 * 28), nn.Tanh())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
