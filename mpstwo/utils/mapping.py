import math
from typing import Any, Callable, Protocol, Union

import torch


class Map(Protocol):
    def __call__(self, input_data: Any) -> torch.Tensor:
        ...

    invert: Callable[..., torch.Tensor]


class OneHotMap:
    """Simple one-hot mapping.

    Keyword arguments:
    dims -- the number of dimensions
    """

    def __init__(self, dims: int) -> None:
        self.dims = dims

    def __call__(self, input_data: torch.Tensor) -> torch.Tensor:
        v = [(input_data == k) * 1.0 for k in range(self.dims)]
        return torch.cat(v, dim=-1)

    def invert(self, input_data: torch.Tensor) -> torch.Tensor:
        c_input = torch.argmax(input_data, dim=-1)
        return torch.arange(self.dims, device=c_input.device)[c_input]


class MultiOneHotMap:
    """One-hot mapping for mutliple categorical variables.

    This class creates one-hot mappings for combinations of multiple
    categorical variables.

    Keyword arguments:
    nums -- list of number of categories for each variable
    """

    def __init__(self, nums: list[int]) -> None:
        self.nums = nums

    def __call__(self, input_data: torch.Tensor) -> torch.Tensor:
        args = []
        in_str = ""
        out_str = ""
        for m, num in enumerate(self.nums):
            idx = chr(97 + m)
            in_str += f",...{idx}"
            out_str += f"{idx}"
        args.append(in_str[1:] + "->..." + out_str)
        for m, num in enumerate(self.nums):
            v = [(input_data[..., m] == k) * 1.0 for k in range(num)]
            arr = torch.stack(v, dim=-1)
            args.append(arr)
        return torch.einsum(*args).flatten(start_dim=-len(self.nums))

    def invert(self, input_data: torch.Tensor) -> torch.Tensor:
        c_input = torch.argmax(input_data, dim=-1)
        return self.invert_argmax(c_input)

    def invert_argmax(self, input_data: torch.Tensor) -> torch.Tensor:
        c_input = input_data.clone()
        inverted = torch.zeros(
            input_data.shape + (len(self.nums),), device=c_input.device
        )
        for m, num in reversed(list(enumerate(self.nums))):
            inverted[..., m] = c_input % num
            c_input = torch.div(c_input, num, rounding_mode="floor")
        return inverted


class FixedOneHotMap:
    """One-hot mapping with fixed positions.

    This class defines a one-hot mapping where the order of the given elements
    is fixed and the location of the element in the list corresponds to
    the location of the `1' in the one-hot vector.

    Keyword arguments:
    possibilities -- values which will be mapped
    """

    def __init__(self, possibilities) -> None:
        self.nums = possibilities

    def __call__(self, input_data: torch.Tensor) -> torch.Tensor:
        v = [(input_data == k) * 1.0 for k in self.nums]
        return torch.cat(v, dim=-1)

    def invert(self, input_data: torch.Tensor) -> torch.Tensor:
        c_input = torch.argmax(input_data, dim=-1)
        return torch.tensor(self.nums, device=c_input.device)[c_input]


class FourierMap:
    """Fourier mapping with variable dimensionality.

    This class creates fourier feature maps where the size of the output
    vector can be chosen for a continuous variable between 0 and 1.

    Keywords arguments:
    dim -- dimensionality of output vector
    transform -- function which maps input to [0, 1]
    """

    def __init__(self, dim=2, transform=None) -> None:
        assert dim % 2 == 0
        self.dim = dim
        if transform is None:
            self.transform = lambda x: x
        else:
            self.transform = transform

    def __call__(self, input_data: Union[float, torch.Tensor]) -> torch.Tensor:
        x = self.transform(input_data)
        r = self.dim // 2
        v = torch.cat(
            [
                torch.exp(-1j * 2 * math.pi * k * x) / math.sqrt(self.dim)
                for k in range(-r, r + 1)
                if k != 0
            ],
            dim=-1,
        )
        return v

    def invert(
        self, input_data: Union[float, torch.Tensor], output_data: torch.Tensor
    ) -> torch.Tensor:
        x = self.transform(input_data)
        r = self.dim // 2
        v = torch.stack(
            [torch.exp(1j * 2 * math.pi * k * x) for k in range(-r, r + 1) if k != 0],
            dim=0,
        )
        return output_data @ v

    def pdf(self, input_data: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
        if input_data.shape[-1] != 1:
            input_data = input_data.unsqueeze(-1)
        v = self(input_data)
        prob = torch.einsum("...i,...ij,...j->...", v.conj(), rho, v)
        if torch.all(torch.imag(prob) < 1e-6):
            return torch.real(prob)
        raise ValueError(f"PDF returns complex number: {prob}")

    def cdf(self, b: float, rho: torch.Tensor) -> torch.Tensor:
        r = self.dim // 2
        x = torch.tensor([0, b])
        keq = torch.diag(rho).sum() * (x[1] - x[0])

        kneq = 0 + 0j
        for m in range(-r, r + 1):
            for n in range(-r, r + 1):
                if m != n:
                    kneq += (
                        -1j
                        / (n - m)
                        / (self.dim**2)
                        * rho[m, n]
                        * (
                            torch.exp(1j * 2 * math.pi * (n - m) * x[1])
                            - torch.exp(1j * 2 * math.pi * (n - m) * x[0])
                        )
                    )
        return keq + kneq

    def exp(self, rho: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class NonPeriodic2DMap:
    """Non-periodic 2D mapping.

    This class creates feature mappings for a continuous variable
    between 0 and 1 as in Stoudenmire and Schwab (2017).

    Keywords arguments:
    transform -- function which maps input to [0, 1]
    """

    def __init__(self, transform=None) -> None:
        if transform is None:
            self.transform = lambda x: x
        else:
            self.transform = transform

    def __call__(self, input_data: Union[float, torch.Tensor]) -> torch.Tensor:
        x = self.transform(input_data)
        v = torch.cat(
            [
                torch.exp(1j * 3 * math.pi / 2 * x) * torch.cos(math.pi / 2 * x),
                torch.exp(-1j * 3 * math.pi / 2 * x) * torch.sin(math.pi / 2 * x),
            ],
            dim=-1,
        )
        return v

    def invert(
        self, input_data: Union[float, torch.Tensor], output_data: torch.Tensor
    ) -> torch.Tensor:
        x = self.transform(input_data)
        v = torch.stack(
            [
                torch.exp(-1j * 3 * math.pi / 2 * x) * torch.cos(math.pi / 2 * x),
                torch.exp(1j * 3 * math.pi / 2 * x) * torch.sin(math.pi / 2 * x),
            ],
            dim=0,
        )
        return output_data @ v

    def pdf(self, input_data: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
        if input_data.shape[-1] != 1:
            input_data = input_data.unsqueeze(-1)
        v = self(input_data)
        prob = torch.einsum("...i,...ij,...j->...", v.conj(), rho, v)
        if torch.all(torch.imag(prob) < 1e-6):
            return torch.real(prob)
        raise ValueError(f"PDF returns complex number: {prob}")

    def cdf(self, b: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
        upper = self.transform(b)
        x = torch.stack([torch.zeros_like(upper), upper], dim=0).to(rho)
        keq = torch.zeros_like(x).to(rho)
        for m in [-1, 1]:
            keq += (
                rho[m, m]
                * (
                    math.pi * (m * (2 * x - 1) - 1)
                    + 2 * torch.cos(math.pi / 2 * m * (2 * x - 1))
                )
                / 4
                / math.pi
                / m
            )

        kneq = torch.zeros_like(x).to(rho)
        for m in [-1, 1]:
            for n in [-1, 1]:
                if m != n:
                    kneq += (
                        rho[m, n]
                        / 8
                        / math.pi
                        * torch.exp(1j * math.pi * (m - n) * x)
                        * (
                            -1
                            / (m - n)
                            * (
                                (-2 + torch.exp(1j * math.pi * (m - n) * x))
                                * math.sin(math.pi / 4 * (m - n))
                                + 1j
                                * (2 + torch.exp(1j * math.pi * (m - n) * x))
                                * math.cos(math.pi / 4 * (m - n))
                            )
                            - (
                                2
                                * torch.exp(1j * math.pi / 2 * (m - n) * x)
                                * (
                                    (m + n)
                                    * torch.cos(math.pi / 4 * (1 - 2 * x) * (m + n))
                                    + 3j
                                    * (m - n)
                                    * torch.sin(math.pi / 4 * (1 - 2 * x) * (m + n))
                                )
                            )
                            / ((m - 2 * n) * (2 * m - n))
                        )
                    )
        prob = keq[1] + kneq[1] - keq[0] + kneq[0]
        if torch.all(torch.imag(prob) < 1e-6):
            return torch.real(prob)
        raise ValueError(f"CDF returns complex number: {prob}")

    def exp(self, rho: torch.Tensor) -> torch.Tensor:
        x = torch.tensor([0, 1])
        part = torch.zeros(2).to(rho)
        for m in [-1, 1]:
            part += (
                rho[m, m]
                / 16
                / math.pi**2
                / m**2
                * (
                    math.pi**2 * (m**2 * (4 * x**2 - 1) - 2 * m - 1)
                    - 8 * torch.sin(math.pi / 2 * m * (2 * x - 1))
                    + 8 * math.pi * m * x * torch.cos(math.pi / 2 * m * (2 * x - 1))
                )
            )
        for m in [-1, 1]:
            for n in [-1, 1]:
                if m != n:
                    part += (
                        rho[m, n]
                        / 16
                        / math.pi**2
                        * torch.exp(1j * math.pi * (m - n) * x)
                        * (
                            1
                            / (m - n) ** 2
                            * (
                                (
                                    torch.exp(1j * math.pi * (m - n) * x)
                                    * (-2j * math.pi * (m - n) * x + 1)
                                    - 4j * math.pi * (m - n) * x
                                    + 4
                                )
                                * math.cos(math.pi / 4 * (m - n))
                                - (
                                    torch.exp(1j * math.pi * (m - n) * x)
                                    * (2 * math.pi * (m - n) * x + 1j)
                                    - 4 * math.pi * (m - n)
                                    - 4j
                                )
                                * math.sin(math.pi / 4 * (m - n))
                            )
                            - (
                                4
                                * torch.exp(1j / 2 * math.pi * (m - n) * x)
                                * (
                                    (m + n)
                                    * (
                                        2 * math.pi * m**2 * x
                                        + m * (-5 * math.pi * n * x + 3j)
                                        + n * (2 * math.pi * n * x - 3j)
                                    )
                                    * torch.cos(math.pi / 4 * (1 - 2 * x) * (m + n))
                                    + -1j
                                    * (
                                        6 * math.pi * m**3 * x
                                        + m**2 * (-21 * math.pi * n * x + 5j)
                                        + m * n * (21 * math.pi * n * x - 8j)
                                        + n**2 * (-6 * math.pi * n * x + 5j)
                                    )
                                    * torch.sin(math.pi / 4 * (1 - 2 * x) * (m + n))
                                )
                            )
                            / ((m - 2 * n) ** 2 * (n - 2 * m) ** 2)
                        )
                    )
        exp = part[1] - part[0]
        if torch.all(torch.imag(exp) < 1e-6):
            return torch.real(exp)
        raise ValueError(f"EXP returns complex number: {exp}")


class RealFourierMap:
    """Fourier mapping without imaginary components with variable dimensionality.

    This class creates fourier feature maps where the size of the output
    vector can be chosen. Complex components have been split up so that
    the vector contains only real components.

    Keywords arguments:
    dim -- dimensionality of output vector
    """

    def __init__(self, dim=2) -> None:
        assert dim % 2 == 0
        self.dim = dim

    def __call__(self, input_data: torch.Tensor) -> torch.Tensor:
        r = self.dim // 2
        v = [
            torch.cos(2 * math.pi / 1.8 * k * (input_data + 1.2)) / math.sqrt(r)
            for k in range(1, r + 1)
        ] + [
            torch.sin(2 * math.pi / 1.8 * k * (input_data + 1.2)) / math.sqrt(r)
            for k in range(1, r + 1)
        ]
        embedded_data = torch.cat(v, dim=-1)
        return embedded_data

    def invert(self, x: torch.Tensor, output_data: torch.Tensor) -> torch.Tensor:
        r = self.dim // 2
        c = output_data[..., :r] + 1j * output_data[..., r:]
        v = [torch.cos(2 * math.pi / 1.8 * k * (x + 1.2)) for k in range(1, r + 1)] + [
            -torch.sin(2 * math.pi / 1.8 * k * (x + 1.2)) for k in range(1, r + 1)
        ]
        v = torch.stack(v, dim=0)
        inverted = c @ (v[:r] + 1j * v[r:])
        return inverted

    def pdf(self, x: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
        r = self.dim // 2
        v = [
            torch.cos(2 * math.pi / 1.8 * k * (x + 1.2)) / math.sqrt(r)
            for k in range(1, r + 1)
        ] + [
            torch.sin(2 * math.pi / 1.8 * k * (x + 1.2)) / math.sqrt(r)
            for k in range(1, r + 1)
        ]
        vec1 = torch.stack(v, dim=-1)
        v = [
            torch.cos(2 * math.pi / 1.8 * k * (x + 1.2)) / math.sqrt(r)
            for k in range(1, r + 1)
        ] + [
            -torch.sin(2 * math.pi / 1.8 * k * (x + 1.2)) / math.sqrt(r)
            for k in range(1, r + 1)
        ]
        vec2 = torch.stack(v, dim=-1)
        return torch.einsum("...i,...ij,...j->...", vec2, rho, vec1)

    def cdf(self, b: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
        r = self.dim // 2
        x = torch.tensor([-1.2, b])
        part = torch.zeros(2).to(rho)
        for m in range(1, r + 1):
            for n in range(1, r + 1):
                if m == n:
                    v1 = torch.zeros(2 * r).to(rho)
                    v2 = torch.zeros(2 * r).to(rho)
                    v1[m - 1] = 1
                    v2[r + m - 1] = 1
                    k = 2 * math.pi / 1.8 * m
                    s = torch.sin(2 * k * (x + 1.2))
                    c = torch.cos(2 * k * (x + 1.2))
                    part += (
                        1
                        / 4
                        / k
                        * (2 * k * (x + 1.2) + s)
                        * torch.einsum("i,ij,j", v1, rho, v1)
                    )
                    part += (
                        1
                        / 4
                        / k
                        * (2 * k * (x + 1.2) - s)
                        * torch.einsum("i,ij,j", v2, rho, v2)
                    )
                    part += 1 / 4 / k * -c * torch.einsum("i,ij,j", v2, rho, v1)
                    part += 1 / 4 / k * -c * torch.einsum("i,ij,j", v1, rho, v2)
                else:
                    v1 = torch.zeros(2 * r).to(rho)
                    v2 = torch.zeros(2 * r).to(rho)
                    v1p = torch.zeros(2 * r).to(rho)
                    v2p = torch.zeros(2 * r).to(rho)
                    v1[m - 1] = 1
                    v2[r + m - 1] = 1
                    v1p[n - 1] = 1
                    v2p[r + n - 1] = 1
                    c = 2 * math.pi / 1.8
                    km = m - n
                    kp = m + n
                    sm = torch.sin(c * km * (x + 1.2)) / km
                    sp = torch.sin(c * kp * (x + 1.2)) / kp
                    cm = torch.cos(c * km * (x + 1.2)) / km
                    cp = torch.cos(c * kp * (x + 1.2)) / kp
                    part += 1 / 2 / c * (sm + sp) * torch.einsum("i,ij,j", v1, rho, v1p)
                    part += 1 / 2 / c * (sm - sp) * torch.einsum("i,ij,j", v2, rho, v2p)
                    part += (
                        1 / 2 / c * -(cm + cp) * torch.einsum("i,ij,j", v2, rho, v1p)
                    )
                    part += 1 / 2 / c * (cm - cp) * torch.einsum("i,ij,j", v1, rho, v2p)
        return part[1] - part[0]

    def exp(self, rho: torch.Tensor) -> torch.Tensor:
        r = self.dim // 2
        x = torch.tensor([-1.2, 0.6])
        part = torch.zeros(2)
        for m in range(1, r + 1):
            for n in range(1, r + 1):
                if m == n:
                    v1 = torch.zeros(2 * r)
                    v2 = torch.zeros(2 * r)
                    v1[m - 1] = 1
                    v2[r + m - 1] = 1
                    k = 2 * math.pi / 1.8 * m
                    s = torch.sin(2 * k * (x + 1.2))
                    c = torch.cos(2 * k * (x + 1.2))
                    part += (
                        1
                        / 8
                        / k**2
                        * (2 * k * x * (s + k * x) + c)
                        * torch.einsum("i,ij,j", v1, rho, v1)
                    )
                    part += (
                        1
                        / 8
                        / k**2
                        * -(2 * k * x * (s - k * x) + c)
                        * torch.einsum("i,ij,j", v2, rho, v2)
                    )
                    part += (
                        1
                        / 8
                        / k**2
                        * (s - 2 * k * x * c)
                        * torch.einsum("i,ij,j", v2, rho, v1)
                    )
                    part += (
                        1
                        / 8
                        / k**2
                        * (s - 2 * k * x * c)
                        * torch.einsum("i,ij,j", v1, rho, v2)
                    )
                else:
                    v1 = torch.zeros(2 * r)
                    v2 = torch.zeros(2 * r)
                    v1p = torch.zeros(2 * r)
                    v2p = torch.zeros(2 * r)
                    v1[m - 1] = 1
                    v2[r + m - 1] = 1
                    v1p[n - 1] = 1
                    v2p[r + n - 1] = 1
                    b = 2 * math.pi / 1.8
                    km = m - n
                    kp = m + n
                    sm = torch.sin(b * km * (x + 1.2)) / km
                    sp = torch.sin(b * kp * (x + 1.2)) / kp
                    cm = torch.cos(b * km * (x + 1.2)) / km
                    cp = torch.cos(b * kp * (x + 1.2)) / kp
                    part += (
                        1
                        / 2
                        / b**2
                        * (b * x * sm + b * x * sp + cm / km + cp / kp)
                        * torch.einsum("i,ij,j", v1, rho, v1p)
                    )
                    part += (
                        1
                        / 2
                        / b**2
                        * (b * x * sm - b * x * sp + cm / km - cp / kp)
                        * torch.einsum("i,ij,j", v2, rho, v2p)
                    )
                    part += (
                        1
                        / 2
                        / b**2
                        * (sm / km + sp / kp - b * x * cm - b * x * cp)
                        * torch.einsum("i,ij,j", v2, rho, v1p)
                    )
                    part += (
                        1
                        / 2
                        / b**2
                        * (-sm / km + sp / kp + b * x * cm - b * x * cp)
                        * torch.einsum("i,ij,j", v1, rho, v2p)
                    )
        return part[1] - part[0]
