from __future__ import annotations

import logging
import math
from os import PathLike
from typing import Optional, Sequence

import torch
from typing_extensions import Self  # move to typing in Python 3.11

from mpstwo.data.datastructs import TensorDict
from mpstwo.model.optimizers import Optimizer
from mpstwo.utils.distributions import (
    Q_obs,
    Q_ogoi,
    Q_oi,
    Q_state,
    likelihood,
    marginal_prob,
    prob_given_seq,
)
from mpstwo.utils.operations import contract, normalize

logger = logging.getLogger(__name__)


class MPSTwo:
    """This class implements a matrix product state which represents an n-form
    with n action inputs and n observation inputs.
    """

    def __init__(
        self,
        physical_legs: int,
        *,
        feature_dim_obs: int = 2,
        feature_dim_act: int = 2,
        bond_dim: int = 2,
        init_mode: str = "random",
        init_std: float = 0.1,
        adaptive_mode: bool = True,
        min_bond: int = 2,
        max_bond: int = 300,
        cutoff: float = 0.01,
        dtype: str | torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
    ) -> None:
        self.physical_legs = physical_legs
        self.feature_dim_obs = feature_dim_obs
        self.feature_dim_act = feature_dim_act

        self.adaptive_mode = adaptive_mode
        self.min_bond = min_bond
        self.max_bond = max_bond
        self.cutoff = cutoff

        self.dtype = eval(dtype) if isinstance(dtype, str) else dtype
        self.device = device
        self._init_mode = init_mode
        self._bond_dim = bond_dim
        self._init_std = init_std
        self._init_matrices(init_mode, bond_dim, init_std)

        # initialize MPS in right canonical gauge
        self.right_canonical()

    def _init_matrices(
        self, init_mode: str, init_bond_dim: int, init_std: float
    ) -> None:
        tensor_shape = (
            self.physical_legs,
            init_bond_dim,
            self.feature_dim_act,
            self.feature_dim_obs,
            init_bond_dim,
        )
        random_tensor = torch.randn(tensor_shape, dtype=self.dtype, device=self.device)

        if "random_eye" in init_mode:
            # normal distribution on identity matrix
            eye_shape = (1, init_bond_dim, 1, 1, init_bond_dim)
            eye_tensor = torch.eye(
                init_bond_dim, init_bond_dim, dtype=self.dtype, device=self.device
            )
            eye_tensor = eye_tensor.view(eye_shape).expand(tensor_shape)
            tensor = eye_tensor + init_std * random_tensor

        elif "positive" in init_mode:
            # half-normal distribution
            if not random_tensor.is_complex():
                tensor = init_std * random_tensor.abs()
            else:
                tensor = init_std * torch.complex(
                    random_tensor.real.abs(), random_tensor.imag
                )

        else:
            # normal distribution
            tensor = init_std * random_tensor

        self.matrices = list(torch.unbind(tensor))
        self.matrices[0] = self.matrices[0][0].unsqueeze(0)
        self.matrices[-1] = self.matrices[-1][..., -1].unsqueeze(-1)

    def right_canonical(self) -> None:
        for i in range(len(self.matrices) - 1, 0, -1):
            s = self.matrices[i].shape
            q, r = torch.linalg.qr(
                self.matrices[i].reshape((s[0], math.prod(s[1:]))).T, mode="reduced"
            )
            self.matrices[i] = q.T.reshape(s)
            self.matrices[i - 1] = contract(self.matrices[i - 1], r.T)
        self.matrices[0] = normalize(self.matrices[0])

    def left_canonical(self) -> None:
        for i in range(len(self.matrices) - 1):
            s = self.matrices[i].shape
            q, r = torch.linalg.qr(
                self.matrices[i].reshape((math.prod(s[:-1]), s[-1])), mode="reduced"
            )
            self.matrices[i] = q.reshape(s)
            self.matrices[i + 1] = contract(r, self.matrices[i + 1])
        self.matrices[-1] = normalize(self.matrices[-1])

    def mixed_canonical(self, idx: int) -> None:  # not well-tested
        for i in range(idx):
            s = self.matrices[i].shape
            q, r = torch.linalg.qr(
                self.matrices[i].reshape((math.prod(s[:-1]), s[-1])), mode="reduced"
            )
            self.matrices[i] = q.reshape(s)
            self.matrices[i + 1] = contract(r, self.matrices[i + 1])
        for i in range(len(self.matrices) - 1, idx, -1):
            s = self.matrices[i].shape
            q, r = torch.linalg.qr(
                self.matrices[i].reshape((s[0], math.prod(s[1:]))).T, mode="reduced"
            )
            self.matrices[i] = q.T.reshape(s)
            self.matrices[i - 1] = contract(self.matrices[i - 1], r.T)
        self.matrices[idx] = normalize(self.matrices[idx])

    def __call__(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        """Returns the probablity of a given set of observations and actions."""
        return self.get_prob(obs, act)

    def get_prob(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        """Calculate the probablity of a set of observations and actions."""
        return torch.abs(self.get_psi(obs, act)) ** 2

    def get_psi(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        """Calculate the probability amplitude of a set of observations and actions."""
        left = torch.ones((1, 1), dtype=self.dtype, device=self.device)
        for i, m in enumerate(self.matrices):
            left = torch.einsum("bl,ba,bo,laor->br", left, act[:, i], obs[:, i], m)
        return left.squeeze(-1)

    def update(
        self, observations: torch.Tensor, actions: torch.Tensor, optimizer: Optimizer
    ) -> None:
        """Update the tensors based on a sequence of observations and actions."""
        self._init_cumulants(0, observations, actions)

        # right sweep until second to last tensor (last bond)
        for i in range(0, self.physical_legs - 1):
            self._train_bond(i, observations, actions, True, optimizer)

        # left sweep starting at last bond
        for i in range(self.physical_legs - 2, -1, -1):
            self._train_bond(i, observations, actions, False, optimizer)

    def _train_bond(
        self,
        i: int,
        observations: torch.Tensor,
        actions: torch.Tensor,
        going_right: bool,
        optimizer: Optimizer,
    ) -> None:
        """Training on current bond

        Arguments:
        i -- index of current bond
        going_right -- whether we're going right
        obs -- observation sequence
        act -- action sequence
        lr -- learning rate for SGD
        """
        obs = observations[:, i : i + 2]
        act = actions[:, i : i + 2]
        left, right = self.cumulants[i : i + 2]

        # SGD on merged tensor
        updated_matrix = optimizer(
            self.matrices[i], self.matrices[i + 1], obs, act, left, right, going_right
        )

        if optimizer.two_site:  # SVD
            self.matrices[i], self.matrices[i + 1] = self.rebuild_bond(
                updated_matrix,
                going_right,
                self.matrices[i + 1].shape[0],
                keep_bond_dim=not self.adaptive_mode,
            )
        else:  # QR
            s = updated_matrix.shape
            if going_right:
                q, r = torch.linalg.qr(
                    updated_matrix.reshape((math.prod(s[:-1]), s[-1])), mode="reduced"
                )
                self.matrices[i] = q.reshape(s)
                self.matrices[i + 1] = contract(r, self.matrices[i + 1])
            else:
                q, r = torch.linalg.qr(
                    updated_matrix.reshape((s[0], math.prod(s[1:]))).T, mode="reduced"
                )
                self.matrices[i + 1] = q.T.reshape(s)
                self.matrices[i] = contract(self.matrices[i], r.T)

        # adjust cumulants
        updated_cumulant = self._update_cumulants(
            self.matrices[i], self.matrices[i + 1], obs, act, left, right, going_right
        )
        if (i != 0 or going_right) and (i != self.physical_legs - 2 or not going_right):
            self.cumulants[i + int(going_right)] = updated_cumulant

    @staticmethod
    def get_psi_cumulants(
        merged_matrix: torch.Tensor,
        obs: torch.Tensor,
        act: torch.Tensor,
        left: torch.Tensor,
        right: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate psi on the training sequence using cumulants"""
        return torch.einsum(
            "bl,ba,bo,laocpr,bc,bp,br->b",
            left,
            act[:, 0],
            obs[:, 0],
            merged_matrix,
            act[:, 1],
            obs[:, 1],
            right,
        )

    @staticmethod
    def get_psi_prime_cumulants(
        obs: torch.Tensor,
        act: torch.Tensor,
        left: torch.Tensor,
        right: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate psi on the training sequence using cumulants"""
        return torch.einsum(
            "bl,ba,bo,bc,bp,br->blaocpr",
            left,
            act[:, 0],
            obs[:, 0],
            act[:, 1],
            obs[:, 1],
            right,
        )

    def rebuild_bond(
        self,
        merged_matrix: torch.Tensor,
        going_right: bool,
        bond_dim: int,
        keep_bond_dim=False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """SVD decomposition with bond dimension selection

        Arguments:
        merged_matrix -- the matrix to be decomposed
        bond_dim -- the original bond dimension of the merged matrices
        going_right -- whether it is a right sweep

        Keyword arguments:
        keep_bond_dim -- whether the previous bond dimension should be kept
        """
        left_dims = merged_matrix.shape[:3]
        right_dims = merged_matrix.shape[-3:]
        U, s, Vh = torch.linalg.svd(
            merged_matrix.reshape((math.prod(left_dims), math.prod(right_dims)))
        )

        if s[0] <= 0.0:
            logging.error(
                "Error: merged_matrix is all-zero.\nPlease tune learning rate."
            )
            raise FloatingPointError("merged_matrix trained to all-zero.")

        if keep_bond_dim:
            thr = bond_dim
        else:
            bdmax = min(self.max_bond, s.shape[0])
            thr = self.min_bond
            while thr < bdmax and s[thr] >= self.cutoff * s[0]:
                thr += 1

        S = torch.diag(s[:thr]).to(merged_matrix)
        U = U[:, :thr]
        Vh = Vh[:thr, :]

        if going_right:
            Vh = S @ Vh
            Vh = normalize(Vh)
        else:
            U = U @ S
            U = normalize(U)

        left = U.reshape(left_dims + (thr,))
        right = Vh.reshape((thr,) + right_dims)
        return left, right

    def _init_cumulants(self, idx: int, obs: torch.Tensor, act: torch.Tensor) -> None:
        """Initialize a cache for left environments and right environments

        During the training phase, it will be kept unchanged that:
        1) len(cumulants)== physical_legs
        2) cumulants[0]  == ones((n_sample, 1))
        3) cumulants[-1] == ones((n_sample, 1))
        4) cumulants[j]  == if 0<j<=k: A(0)...A(j-1)
                            elif k<j<physical_legs-1: A(j+1)...A(physical_legs-1)

        Cumulants work slightly differently from the original implementation.
        Originally, they were stored for the entire dataset at once.
        This is impractical for large datasets for obvious reasons.
        In this implementation, this method must be called again for each batch.

        Arguments:
        obs -- observation sequence
        act -- action sequence
        idx -- idx of current bond
        """
        left = [torch.ones((1, 1), dtype=self.dtype, device=self.device)]
        for i, m in enumerate(self.matrices[:idx]):
            left.append(
                normalize(
                    torch.einsum("bl,laor,ba,bo->br", left[-1], m, act[:, i], obs[:, i])
                )
            )
        right = [torch.ones((1, 1), dtype=self.dtype, device=self.device)]
        for i, m in reversed(list(enumerate(self.matrices))[idx + 2 :]):
            right.insert(
                0,
                normalize(
                    torch.einsum("laor,ba,bo,br->bl", m, act[:, i], obs[:, i], right[0])
                ),
            )
        self.cumulants = left + right

    @staticmethod
    def _update_cumulants(
        m1: torch.Tensor,
        m2: torch.Tensor,
        obs: torch.Tensor,
        act: torch.Tensor,
        left: torch.Tensor,
        right: torch.Tensor,
        going_right: bool,
    ) -> torch.Tensor:
        """Update cumulants after rebuilding bonds

        The bond has been rebuilt, so it matters whether we were going right.
        """
        if going_right:
            return normalize(
                torch.einsum("bl,laor,ba,bo->br", left, m1, act[:, 0], obs[:, 0])
            )
        else:
            return normalize(
                torch.einsum("laor,ba,bo,br->bl", m2, act[:, 1], obs[:, 1], right)
            )

    def get_sv(self, bond: int) -> torch.Tensor:
        self.mixed_canonical(bond)
        merged_matrix = contract(self.matrices[bond], self.matrices[bond + 1])
        left_dims = merged_matrix.shape[:3]
        right_dims = merged_matrix.shape[-3:]
        _, s, _ = torch.linalg.svd(
            merged_matrix.reshape((math.prod(left_dims), math.prod(right_dims)))
        )
        return s

    def truncate(
        self,
        *,
        cutoff: Optional[float] = None,
        dim: Optional[int | Sequence[int]] = None,
    ) -> None:
        assert cutoff is None != dim is None, "cutoff xor dim must be filled in"

        self.right_canonical()

        bond_dim = [m.shape[-1] for m in self.matrices] + [1]
        if cutoff is not None:
            self.cutoff = cutoff
        elif dim is not None:
            if isinstance(dim, int):
                bond_dim = [dim for _ in range(self.physical_legs)]
                bond_dim[-1] = 1
            else:
                assert len(dim) == self.physical_legs
                assert dim[-1] == 1, "last bond must be 1"
                bond_dim = list(dim)

        # perform svd with given cutoff or bond_dim
        for i, m in enumerate(self.matrices[1:]):
            merged_matrix = contract(self.matrices[i], m)
            left, right = self.rebuild_bond(
                merged_matrix,
                going_right=True,
                bond_dim=bond_dim[i],
                keep_bond_dim=(dim is not None),
            )
            self.matrices[i] = left
            self.matrices[i + 1] = right

    @property
    def bond_dim(self):
        return [m.shape[-1] for m in self.matrices]

    def norm(self) -> torch.Tensor:
        left = torch.ones((1, 1), dtype=self.dtype, device=self.device)
        for m in self.matrices:
            left = torch.einsum("lm,laor->maor", left, m)
            left = torch.einsum("maor,maos->rs", left, m.conj())
        return left.squeeze(-1).abs()

    def conj(self):
        new_mps = self.clone()
        new_mps.matrices = [m.conj() for m in self.matrices]
        return new_mps

    def __matmul__(self, other):
        if isinstance(other, MPSTwo):
            assert len(self.matrices) == len(other.matrices)
            if self.device != other.device:
                other.to(self.device)
            left = torch.ones((1, 1), dtype=self.dtype, device=self.device)
            for i in range(len(self.matrices)):
                left = torch.einsum("lm,laor->maor", left, self.matrices[i])
                left = torch.einsum("maor,maos->rs", left, other.matrices[i].conj())
            return left.squeeze(-1)
        elif isinstance(other, TensorDict):
            return self.get_psi(other["observation"], other["action"])
        else:
            raise TypeError

    def __rmatmul__(self, other):
        if isinstance(other, TensorDict):
            return self @ other
        else:
            raise TypeError

    def __eq__(self, other: MPSTwo) -> bool:
        if self.physical_legs != other.physical_legs:
            return False
        for m, n in zip(self.matrices, other.matrices):
            if not torch.equal(m, n):
                return False
        return True

    def is_complex(self) -> bool:
        return self.matrices[0].is_complex()

    def to(self, device: str | torch.device) -> Self:
        self.device = device
        self.matrices = [m.to(device) for m in self.matrices]
        try:
            self.cumulants = [c.to(device) for c in self.cumulants]
        except AttributeError:
            pass
        return self

    def clone(self) -> Self:
        model = type(self)(**self.hyperparameters)
        model.matrices = [m.clone() for m in self.matrices]
        try:
            model.cumulants = [c.clone() for c in self.cumulants]
        except AttributeError:
            pass
        return model

    def get_parameters(self) -> int:
        return sum(m.numel() for m in self.matrices)

    @property
    def hyperparameters(self) -> dict:
        p_dict = {
            "physical_legs": self.physical_legs,
            "feature_dim_act": self.feature_dim_act,
            "feature_dim_obs": self.feature_dim_obs,
            "bond_dim": self._bond_dim,
            "init_mode": self._init_mode,
            "init_std": self._init_std,
            "adaptive_mode": self.adaptive_mode,
            "min_bond": self.min_bond,
            "max_bond": self.max_bond,
            "cutoff": self.cutoff,
            "dtype": self.dtype,
            "device": self.device,
        }
        return p_dict

    def save(self, path: str | PathLike) -> None:
        torch.save(self, path)

    @classmethod
    def load_full(
        cls, path: str | PathLike, device: Optional[str | torch.device] = None
    ) -> Self:
        return torch.load(path, map_location=device)

    def prob_given_seq(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        idx: int = 1,
        canonical: Optional[str] = None,
    ) -> tuple[torch.Tensor, str]:
        """Compute the probability of a given sequence in addition to open legs

        P(o_0, a_0, ..., o_n, a_n, ..., o_n+idx, a_n+idx)

        Arguments:
            obs -- observation sequence to contract with
            act -- action sequence to contract with

        Keyword Arguments:
            idx -- number of open matrices after sequence (default: {1})
            canonical -- canonical form of MPS (default: {None})

        Returns:
            probs -- probability of given sequence plus open legs
            istr -- corresponding label string of the remaining legs
                Label string has the form `aobpcq...'.
        """
        return prob_given_seq(self, obs, act, idx=idx, canonical=canonical)

    def likelihood(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        idx: int = 1,
        canonical: Optional[str] = None,
    ) -> torch.Tensor:
        """Q(o_i | s_i, a_<=i)"""
        return likelihood(self, obs, act, idx=idx, canonical=canonical)

    def Q_obs(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        idx: int = 1,
        canonical: Optional[str] = None,
    ) -> torch.Tensor:
        """Q(o_i | a_<=i)"""
        return Q_obs(self, obs, act, idx=idx, canonical=canonical)

    def Q_state(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        idx: int = 1,
        canonical: Optional[str] = None,
    ) -> torch.Tensor:
        """Q(s_i | a<=i)

        Equals 1 for idx=1.
        """
        return Q_state(self, obs, act, idx=idx, canonical=canonical)

    def marginal_prob(
        self, indices: list[int], canonical: Optional[str] = None
    ) -> tuple[torch.Tensor, str]:
        """Compute the marginal probability of actions and observations

        P(o_idx0, a_idx0, ..., o_idxn, a_idxn)

        Arguments:
            indices -- indices of open matrices

        Keyword Arguments:
            canonical -- canonical form of MPS (default: {None})

        Returns:
            probs -- marginal probability over open legs
            istr -- corresponding label string of the remaining legs
                Label string has the form `aobpcq...'.
        """
        return marginal_prob(self, indices, canonical=canonical)

    def Q_oi(self, i: int, canonical: Optional[str] = None) -> torch.Tensor:
        """Q(o_i)"""
        return Q_oi(self, i, canonical=canonical)

    def Q_ogoi(self, i: int, canonical: Optional[str] = None) -> torch.Tensor:
        """Q(o_i+1 | o_i)"""
        return Q_ogoi(self, i, canonical=canonical)
