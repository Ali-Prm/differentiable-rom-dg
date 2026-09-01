"""
Nonlinear Manifold ROM architecture for periodic Burgers.

  q = Encoder(U)
  U_pred = Decoder(x_grid, q)
"""

from __future__ import annotations
import torch
import torch.nn as nn
from torch.func import jacfwd, vmap


class Sine(nn.Module):
    def __init__(self, w0: float = 1.0):
        super().__init__()
        self.w0 = float(w0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w0 * x)


_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "elu": nn.ELU,
    "leakyrelu": nn.LeakyReLU,
    "sine": Sine,
}


def _act(name: str) -> nn.Module:
    key = name.lower()
    if key not in _ACTIVATIONS:
        raise ValueError(f"Unknown activation '{name}'. Choose from {list(_ACTIVATIONS)}.")
    return _ACTIVATIONS[key]()


class CNNEncoderNet(nn.Module):
    def __init__(
        self,
        q_dim: int,
        input_nodes: int,
        conv_channels: tuple[int, ...] = (8, 16, 32, 64),
        strides: tuple[int, ...] = (2, 2, 2, 2),
        kernel_size: int = 9,
        activation: str = "elu",
        fc_hidden: int = 128,
        fc_layers: int = 2,
    ):
        super().__init__()
        self.q_dim = q_dim
        self.input_nodes = input_nodes

        layers: list[nn.Module] = []
        in_c = 1
        for out_c, stride in zip(conv_channels, strides):
            layers.append(
                nn.Conv1d(
                    in_c,
                    out_c,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=kernel_size // 2,
                )
            )
            layers.append(_act(activation))
            in_c = out_c

        self.conv_block = nn.Sequential(*layers)

        # Automatically determine feature dimension
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_nodes)
            out_dummy = self.conv_block(dummy)
            feature_dim = out_dummy.view(1, -1).size(1)

        fc_list: list[nn.Module] = []
        in_fc = feature_dim
        for _ in range(fc_layers):
            fc_list.append(nn.Linear(in_fc, fc_hidden))
            fc_list.append(_act(activation))
            in_fc = fc_hidden

        fc_list.append(nn.Linear(in_fc, q_dim))
        self.fc_block = nn.Sequential(*fc_list)

    def forward(self, U: torch.Tensor) -> torch.Tensor:
        x = U.unsqueeze(1)          # (B, 1, Nx)
        x = self.conv_block(x)      # (B, C, L)
        x = x.flatten(start_dim=1)  # (B, C*L)
        return self.fc_block(x)     # (B, q_dim)


class CoordinateDecoderNet(nn.Module):
    def __init__(
        self,
        q_dim: int,
        depth: int = 4,
        width: int = 64,
        activation: str = "tanh",
    ):
        super().__init__()
        self.q_dim = q_dim
        in_dim = 1 + q_dim

        layers: list[nn.Module] = [nn.Linear(in_dim, width), _act(activation)]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(width, width), _act(activation)])
        layers.append(nn.Linear(width, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x_grid: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        B = q.shape[0]
        Nx = x_grid.shape[0]

        x_exp = x_grid[None, :, None].expand(B, Nx, 1)
        q_exp = q[:, None, :].expand(B, Nx, self.q_dim)
        inp = torch.cat([x_exp, q_exp], dim=-1)

        out = self.net(inp.reshape(B * Nx, 1 + self.q_dim))
        return out.reshape(B, Nx)


# class ManifoldROM(nn.Module):
#     def __init__(
#         self,
#         q_dim: int,
#         n_nodes: int,
#         decoder_depth: int = 4,
#         decoder_width: int = 64,
#         decoder_act: str = "tanh",
#         encoder_conv_channels: tuple[int, ...] = (8, 16, 32, 64),
#         encoder_conv_strides: tuple[int, ...] = (2, 2, 2, 2),
#         encoder_kernel_size: int = 9,
#         encoder_activation: str = "elu",
#         encoder_fc_hidden_dim: int = 128,
#         encoder_fc_layers: int = 2,
#     ):
#         super().__init__()
#         self.q_dim = q_dim
#         self.n_nodes = n_nodes

#         self.encoder = CNNEncoderNet(
#             q_dim=q_dim,
#             input_nodes=n_nodes,
#             conv_channels=encoder_conv_channels,
#             strides=encoder_conv_strides,
#             kernel_size=encoder_kernel_size,
#             activation=encoder_activation,
#             fc_hidden=encoder_fc_hidden_dim,
#             fc_layers=encoder_fc_layers,
#         )

#         self.decoder = CoordinateDecoderNet(
#             q_dim=q_dim,
#             depth=decoder_depth,
#             width=decoder_width,
#             activation=decoder_act,
#         )

#     def encode(self, U: torch.Tensor) -> torch.Tensor:
#         return self.encoder(U)

#     def decode(self, x_grid: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
#         return self.decoder(x_grid, q)

#     def decoder_jacobian(
#         self, x_grid: torch.Tensor, q: torch.Tensor
#     ) -> tuple[torch.Tensor, torch.Tensor]:
#         """Compute U_pred: (B, Nx) and J: (B, Nx, q_dim) via batched forward-mode AD."""
#         def _single(q_b: torch.Tensor):
#             u = self.decode(x_grid, q_b.unsqueeze(0)).squeeze(0)
#             return u, u

#         J, U_pred = vmap(jacfwd(_single, has_aux=True))(q)
#         return U_pred, J

#     def forward(
#         self, x_grid: torch.Tensor, U: torch.Tensor
#     ) -> tuple[torch.Tensor, torch.Tensor]:
#         q = self.encode(U)
#         U_pred = self.decode(x_grid, q)
#         return U_pred, q



class TemporalDynamicsNet(nn.Module):
    """
    Feed-forward neural network for latent temporal dynamics.
    
        (q, params) -> q_dot_pred
        
    Injecting the physical parameters is crucial for learning the wave speeds.
    """
    def __init__(
        self,
        q_dim: int,
        param_dim: int,
        depth: int = 3,
        width: int = 64,
        activation: str = "elu"
    ):
        super().__init__()
        
        in_dim = q_dim + param_dim
        layers: list[nn.Module] = [
            nn.Linear(in_dim, width),
            _act(activation),
        ]
        
        for _ in range(depth - 1):
            layers.extend([
                nn.Linear(width, width),
                _act(activation),
            ])
            
        layers.append(nn.Linear(width, q_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, q: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        # Concatenate latent state and physical parameters
        x = torch.cat([q, params], dim=-1)
        return self.net(x)


class ManifoldROM(nn.Module):
    def __init__(
        self,
        q_dim: int,
        n_nodes: int,
        param_dim: int = 2,  # <--- Added parameter dimension
        decoder_depth: int = 4,
        decoder_width: int = 64,
        decoder_act: str = "tanh",
        encoder_conv_channels: tuple[int, ...] = (8, 16, 32, 64),
        encoder_conv_strides: tuple[int, ...] = (2, 2, 2, 2),
        encoder_kernel_size: int = 9,
        encoder_activation: str = "elu",
        encoder_fc_hidden_dim: int = 128,
        encoder_fc_layers: int = 2,
        temporal_depth: int = 3,     # <--- Temporal net config
        temporal_width: int = 64,    # <--- Temporal net config
        temporal_act: str = "elu"    # <--- Temporal net config
    ):
        super().__init__()

        self.q_dim = q_dim
        self.n_nodes = n_nodes

        self.encoder = CNNEncoderNet(
            q_dim=q_dim,
            input_nodes=n_nodes,
            conv_channels=encoder_conv_channels,
            strides=encoder_conv_strides,
            kernel_size=encoder_kernel_size,
            activation=encoder_activation,
            fc_hidden=encoder_fc_hidden_dim,
            fc_layers=encoder_fc_layers,
        )

        self.decoder = CoordinateDecoderNet(
            q_dim=q_dim,
            depth=decoder_depth,
            width=decoder_width,
            activation=decoder_act,
        )
        
        # Instantiate the Temporal Dynamics Network
        self.temporal_net = TemporalDynamicsNet(
            q_dim=q_dim,
            param_dim=param_dim,
            depth=temporal_depth,
            width=temporal_width,
            activation=temporal_act
        )

    def encode(self, U: torch.Tensor) -> torch.Tensor:
            return self.encoder(U)
    
    def decode(self, x_grid: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        return self.decoder(x_grid, q)

    def decoder_jacobian(
        self, x_grid: torch.Tensor, q: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute U_pred: (B, Nx) and J: (B, Nx, q_dim) via batched forward-mode AD."""
        def _single(q_b: torch.Tensor):
            u = self.decode(x_grid, q_b.unsqueeze(0)).squeeze(0)
            return u, u

        J, U_pred = vmap(jacfwd(_single, has_aux=True))(q)
        return U_pred, J

    def forward(
        self, x_grid: torch.Tensor, U: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.encode(U)
        U_pred = self.decode(x_grid, q)
        return U_pred, q