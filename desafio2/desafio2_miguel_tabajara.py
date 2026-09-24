# Miguel Soto Tabajara
import math
import torch

K = 0.15
INCLINACAO = 0.5
E_F2 = K * K * (1 + INCLINACAO ** 2) / 2


def ativacao(x: torch.Tensor) -> torch.Tensor:
    return K * torch.nn.functional.leaky_relu(x, INCLINACAO)


@torch.no_grad()
def inicializar(W: torch.Tensor, b: torch.Tensor,
                fan_in: int, fan_out: int, camada: int, n_camadas: int) -> None:
    if camada == 1:
        desvio = math.sqrt(1.0 / fan_in)
    else:
        desvio = math.sqrt(1.0 / (fan_in * E_F2))
    W.normal_(0.0, desvio)
    b.zero_()
