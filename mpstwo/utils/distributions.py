import logging

import torch

from mpstwo.utils.operations import transfer_normalize

logger = logging.getLogger(__name__)


def kl_divergence(p: torch.Tensor, q: torch.Tensor, dim: int = -1) -> torch.Tensor:
    a = p * torch.log(p / q)
    a[..., (p == 0).expand(a.shape)] = 0
    return torch.sum(a, dim=dim)


def cross_entropy(p: torch.Tensor, q: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return -torch.sum(p * torch.log(q), dim=dim)


def entropy(p: torch.Tensor, dim: int = -1) -> torch.Tensor:
    a = p * torch.log(p)
    a[p == 0] = 0
    return -torch.sum(a, dim=dim)


def prob_given_seq(model, obs, act, idx=1, canonical=None):
    """Compute the probability of a given sequence in addition to open legs

    P(o_0, a_0, ..., o_seq, a_seq, ..., o_seq+idx, a_seq+idx)
    Note that this is only valid for idx >= 1.

    Arguments:
        model -- the MPS
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
    t = act.shape[1]
    left = torch.ones((1, 1, 1), dtype=model.dtype, device=model.device)
    for i, m in enumerate(model.matrices[:t]):
        _m = torch.einsum("LaoR,Ba,Bo->BLR", m, act[:, i], obs[:, i])
        left = torch.einsum("BLM,BLR->BRM", left, _m)
        left = torch.einsum("BRM,BMS->BRS", left, _m.conj())
        left = transfer_normalize(left, idx=0)

    istr = ""
    old_istr = ""
    for j, m in enumerate(model.matrices[t : t + idx]):
        old_istr = istr
        new_istr = chr(ord("a") + j) + chr(ord("o") + j)
        istr += new_istr
        left = torch.einsum(f"B{old_istr}LM,L{new_istr}R->B{istr}RM", left, m)
        left = torch.einsum(f"B{istr}RM,M{new_istr}S->B{istr}RS", left, m.conj())
        left = transfer_normalize(left, idx=0)

    if canonical == "right":
        probs = torch.einsum(f"B{istr}RR->B{istr}", left)
    else:
        right = torch.ones((1, 1), dtype=model.dtype, device=model.device)
        for m in reversed(model.matrices[t + idx :]):
            right = torch.einsum("MaoS,RS->RMao", m.conj(), right)
            right = torch.einsum("LaoR,RMao->LM", m, right)
            right = transfer_normalize(right)
        probs = torch.einsum(f"B{istr}RS,RS->B{istr}", left, right)

    negative_probs = probs.real < 0
    if negative_probs.any():
        logger.warning("Some probabilities were negative.")
        probs[negative_probs] = 0.0

    return probs, istr


def likelihood(model, obs, act, idx=1, canonical=None):
    """Q(o_i | s_i, a_<=i)"""
    probs, istr = prob_given_seq(model, obs, act, idx, canonical)
    probs_g = torch.einsum(f"B{istr}->B{istr[:2*idx-1]}", probs)
    ind = probs_g.nonzero(as_tuple=True)
    probs[ind] /= probs_g.unsqueeze(-1)[ind]
    nind = (probs_g == 0).nonzero(as_tuple=True)
    probs[nind] = 1 / probs.shape[-1]
    return probs


def Q_obs(model, obs, act, idx=1, canonical=None):
    """Q(o_i | a_<=i)"""
    probs, istr = prob_given_seq(model, obs, act, idx, canonical)
    probs = torch.einsum(f"B{istr}->B{istr[::2] + istr[2*idx-1]}", probs)
    probs_g = torch.einsum(f"B{istr[::2] + istr[2*idx-1]}->B{istr[::2]}", probs)
    ind = probs_g.nonzero(as_tuple=True)
    probs[ind] /= probs_g.unsqueeze(-1)[ind]
    nind = (probs_g == 0).nonzero(as_tuple=True)
    probs[nind] = 1 / probs.shape[-1]
    return probs


def Q_state(model, obs, act, idx=1, canonical=None):
    """Q(s_i | a<=i)

    Equals 1 for idx=1.
    """
    probs, istr = prob_given_seq(model, obs, act, idx, canonical)
    probs = torch.einsum(f"B{istr}->B{istr[:-1]}", probs)
    probs_g = torch.einsum(f"B{istr[:-1]}->B{istr[::2]}", probs)
    for i in range(len(istr[: 2 * idx - 1]) - len(istr[::2])):
        probs_g = probs_g.unsqueeze(1 + 2 * i + 1)
    ind = probs.nonzero(as_tuple=True)
    probs[ind] /= probs_g.expand_as(probs)[ind]
    return probs


def Q_o2go1(model, obs, act, idx=2, canonical=None):
    """Q(o_i+1 | o_i, s_i , pi)

    Currently only works for idx = 2.
    """
    probs, istr = prob_given_seq(model, obs, act, idx, canonical)
    probs = torch.einsum(f"B{istr}->B{istr[:2*idx-2] + istr[2*idx-1]}", probs)
    probs /= torch.einsum(
        f"B{istr[:2*idx-2] + istr[2*idx-1]}->B{istr[:2*idx-2]}", probs
    ).unsqueeze(-1)
    return probs


def Q_o2(model, obs, act, idx=2, canonical=None):
    """Q(o_i+1 | s_i , pi)

    Currently only works for idx = 2.
    """
    contr, istr = prob_given_seq(model, obs, act, idx, canonical)
    probs = torch.einsum(f"B{istr}->B{istr[:2*idx-2:2] + istr[2*idx-1]}", contr)
    probs /= torch.einsum(
        f"B{istr[:2*idx-2:2] + istr[2*idx-1]}->B{istr[:2*idx-2:2]}", contr
    ).unsqueeze(-1)
    return probs


def dens_given_seq(model, obs, act, idx=1, canonical=None):
    """Compute the densiry matrix of a given sequence in addition to open legs

    rho(o_0, a_0, ..., o_seq, a_seq, ..., o_seq+idx, a_seq+idx)

    Arguments:
        model -- the MPS
        obs -- observation sequence to contract with
        act -- action sequence to contract with

    Keyword Arguments:
        idx -- number of open matrices after sequence (default: {1})
        canonical -- canonical form of MPS (default: {None})

    Returns:
        probs -- density matrix of given sequence plus open legs
        istr -- corresponding label string of the remaining legs
            Label string has the form `aobpcq...'.
    """
    t = act.shape[1]
    left = torch.ones((1, 1, 1), dtype=model.dtype, device=model.device)
    for i, m in enumerate(model.matrices[:t]):
        left = torch.einsum(
            "BLM,LaoR,Ba,Bo,MbpS,Bb,Bp->BRS",
            left,
            m,
            act[:, i],
            obs[:, i],
            m.conj(),
            act[:, i].conj(),
            obs[:, i].conj(),
        )
        left = transfer_normalize(left, idx=0)

    istr = ""
    old_istr = ""
    for j, m in enumerate(model.matrices[t : t + idx]):
        old_istr = istr
        new_istr1 = chr(ord("a") + j * 2) + chr(ord("o") + j * 2)
        new_istr2 = chr(ord("a") + j * 2 + 1) + chr(ord("o") + j * 2 + 1)
        istr += new_istr1 + new_istr2
        left = torch.einsum(
            f"B{old_istr}LM,L{new_istr1}R,M{new_istr2}S->B{istr}RS", left, m, m.conj()
        )

    if canonical == "right":
        probs = torch.einsum(f"B{istr}RR->B{istr}", left)
    else:
        right = torch.ones((1, 1), dtype=model.dtype, device=model.device)
        for m in reversed(model.matrices[t + idx :]):
            right = torch.einsum("laor,maos,rs->lm", m, m.conj(), right)
            right = transfer_normalize(right)
        probs = torch.einsum(f"B{istr}RS,RS->B{istr}", left, right)

    return probs, istr


def likelihood_(model, obs, act, idx=1, canonical=None):
    """Q(o_i | s_i, a_<=i) calculated through density matrix"""
    dens, istr = dens_given_seq(model, obs, act, idx, canonical)
    ostr = (
        "".join([istr[x : x + 2] * 2 for x in range(0, len(istr[:-4]), 4)])
        + istr[-4:-2]
        + istr[-4]
        + istr[-1]
    )
    nstr = (
        "".join([istr[x : x + 2] for x in range(0, len(istr[:-4]), 4)])
        + istr[-4]
        + istr[-3]
        + istr[-1]
    )
    mstr = (
        "".join([istr[x : x + 2] for x in range(0, len(istr[:-4]), 4)])
        + istr[-4]
        + istr[-3]
        + istr[-3]
    )
    dens = torch.einsum(f"B{ostr}->B{nstr}", dens)
    norm = torch.einsum(f"B{mstr}->B{mstr[:-2]}", dens).unsqueeze(-1).unsqueeze(-1)
    return dens / norm


def marginal_prob(model, indices, canonical=None):
    """Compute the marginal probability of actions and observations

    P(o_idx0, a_idx0, ..., o_idxn, a_idxn)

    Arguments:
        model -- the MPS
        indices -- indices of open matrices

    Returns:
        probs -- marginal probability over open legs
        istr -- corresponding label string of the remaining legs
            Label string has the form `aobpcq...'.
    """
    left = torch.ones((1, 1), dtype=model.dtype, device=model.device)
    istr = ""
    old_istr = ""
    j = 0
    for i, m in enumerate(model.matrices[: max(indices) + 1]):
        if i in indices:
            old_istr = istr
            new_istr = chr(ord("a") + j) + chr(ord("o") + j)
            istr += new_istr
            j += 1
            left = torch.einsum(f"{old_istr}LM,L{new_istr}R->{istr}RM", left, m)
            left = torch.einsum(f"{istr}RM,M{new_istr}S->{istr}RS", left, m.conj())
        else:
            left = torch.einsum(f"{istr}LM,LAOR->{istr}AORM", left, m)
            left = torch.einsum(f"{istr}AORM,MAOS->{istr}RS", left, m.conj())
        left = transfer_normalize(left)

    if canonical == "right":
        probs = torch.einsum(f"{istr}RR->{istr}", left)
    else:
        right = torch.ones((1, 1), dtype=model.dtype, device=model.device)
        for m in reversed(model.matrices[max(indices) + 1 :]):
            right = torch.einsum("MaoS,RS->RMao", m.conj(), right)
            right = torch.einsum("LaoR,RMao->LM", m, right)
            right = transfer_normalize(right)
        probs = torch.einsum(f"{istr}RS,RS->{istr}", left, right)

    return probs, istr


def Q_oi(model, i, canonical=None):
    """Q(o_i)"""
    probs, _ = marginal_prob(model, [i], canonical=canonical)
    probs = torch.einsum("ao->o", probs)
    probs = probs / probs.sum()
    return probs


def Q_ogoi(model, i, canonical=None):
    """Q(o_i+1 | o_i)"""
    probs, _ = marginal_prob(model, [i, i + 1], canonical=canonical)
    probs = torch.einsum("aobp->op", probs)
    probs_g = torch.einsum("op,op->o", probs)
    ind = probs_g.nonzero(as_tuple=True)
    probs[ind] /= probs_g.unsqueeze(-1)[ind]
    return probs
