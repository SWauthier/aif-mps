import pathlib
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from mpstwo.data.datastructs import TensorDict
from mpstwo.envs.wrappers import EnvWrapper

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
    }
)


def visualize_sequence(env: EnvWrapper, sequence: TensorDict):
    env.reset()
    r = env.render()
    if r is not None:
        print(r)
    for i in range(sequence.shape[0]):
        env.s = sequence.observation[i].item()
        env.lastaction = sequence.action[i].item()
        r = env.render()
        if r is not None:
            print(r)


def to_bar(values: list):
    fig, ax = plt.subplots()
    x = range(len(values))
    ax.bar(x, values)
    ax.set_xticks(x, labels=map(str, x))
    plt.close()
    return fig


def to_bar_rgb(values: list) -> np.ndarray:
    fig, ax = plt.subplots()
    x = range(len(values))
    ax.bar(x, values)
    ax.set_xticks(x, labels=map(str, x))
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)  # type: ignore
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    data = data.transpose(2, 0, 1)
    plt.close()
    return data


def tb2pandas(path: str):
    runlog_data = pd.DataFrame({"metric": [], "step": [], "value": []})
    event_acc = EventAccumulator(path)
    event_acc.Reload()
    tags = event_acc.Tags()["scalars"]
    for tag in tags:
        event_list = event_acc.Scalars(tag)
        values = list(map(lambda x: x.value, event_list))
        step = list(map(lambda x: x.step, event_list))
        r = {"metric": [tag] * len(step), "step": step, "value": values}
        r = pd.DataFrame(r)
        runlog_data = pd.concat([runlog_data, r])
    return runlog_data


def plot_relplot(
    A: pd.DataFrame,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    xscale: Optional[str] = None,
    yscale: Optional[str] = None,
    xlim: Optional[tuple[float | None, float | None]] = None,
    ylim: Optional[tuple[float | None, float | None]] = None,
    addhline: Optional[dict | list[dict]] = None,
    legend_ncol: Optional[int] = None,
    show: bool = True,
    save: bool = False,
    save_loc: Optional[pathlib.Path] = None,
    save_name: Optional[str] = None,
    save_suffix: str = ".png",
    **kwargs: Any
) -> None:
    g = sns.relplot(A, palette=sns.color_palette("colorblind"), **kwargs)
    g.figure.subplots_adjust(wspace=0, hspace=0)
    if xlabel is not None:
        g.set_axis_labels(xlabel=xlabel)
    if ylabel is not None:
        g.set_axis_labels(ylabel=ylabel)
    if xscale is not None:
        g.set(xscale=xscale)
    if yscale is not None:
        g.set(yscale=yscale)
    if xlim is not None:
        g.set(xlim=xlim)
    if ylim is not None:
        g.set(ylim=ylim)
    if addhline is not None:
        if isinstance(addhline, dict):
            addhline = [addhline]
        for j, d in enumerate(addhline):
            for i, ax in enumerate(g.figure.axes):
                ax.axhline(**d)
                if "label" in d and i % 2 == 1:
                    ax.text(
                        1.01,
                        d["y"],
                        d["label"],
                        fontsize="xx-small",
                        ha="left",
                        va="bottom" if j == 5 else "center",
                        transform=ax.get_yaxis_transform(),
                        bbox=dict(
                            facecolor="white",
                            alpha=1.0,
                            boxstyle="Square, pad=0.0",
                            edgecolor="none",
                        ),
                    )
    if legend_ncol is not None:
        sns.move_legend(g, "center left", bbox_to_anchor=(0.75, 0.5), ncol=legend_ncol)
    if save:
        path = save_loc if save_loc is not None else pathlib.Path.cwd()
        base = save_name if save_name is not None else "relplot"
        name = base + save_suffix
        i = 0
        while (path / name).exists():
            name = base + "(" + str(i) + ")" + save_suffix
            i += 1
        plt.savefig(
            path / name, dpi=300, facecolor="w", transparent=False, bbox_inches="tight"
        )
    if show:
        plt.show()
    plt.close()


def plot_likelihood(
    A: torch.Tensor,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    xlabelpad: float = 4.0,
    ylabel: Optional[str] = None,
    xlabs: Optional[Sequence[str | int]] = None,
    xrot: Optional[int] = None,
    ylabs: Optional[Sequence[str | int]] = None,
    yrot: Optional[int] = None,
    show: bool = True,
    save: bool = False,
    save_loc: Optional[pathlib.Path] = None,
    save_name: Optional[str] = None,
    save_suffix: str = ".png",
) -> None:
    if A.shape[0] > 4:
        plt.figure(figsize=(3, 10))
    else:
        plt.figure(figsize=(2, 1.5))
    ax = sns.heatmap(
        A,
        cmap=sns.color_palette("YlOrBr", as_cmap=True),
        linewidth=0.0,
        vmin=0,
        vmax=1,
        square=True,
    )
    if xlabs is not None:
        ax.set_xticklabels(xlabs, rotation=xrot)
    if ylabs is not None:
        # ax.set_yticks([y + 0.5 for y in range(A.shape[0])])
        ax.set_yticklabels(ylabs, rotation=yrot)
    if title is not None:
        plt.title(title)
    if xlabel is not None:
        plt.xlabel(xlabel, labelpad=xlabelpad)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if save:
        path = save_loc if save_loc is not None else pathlib.Path.cwd()
        base = save_name if save_name is not None else "likelihood"
        name = base + save_suffix
        i = 0
        while (path / name).exists():
            name = base + "(" + str(i) + ")" + save_suffix
            i += 1
        plt.savefig(
            path / name, dpi=300, facecolor="w", transparent=False, bbox_inches="tight"
        )
    if show:
        plt.show()
    plt.close()


def plot_energy(
    A: torch.Tensor,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    xlabs: Optional[Sequence[str | int]] = None,
    xrot: Optional[int] = None,
    ylim: Optional[int] = None,
    width: float = 2.5,
    height: float = 1.5,
    show: bool = True,
    save: bool = False,
    save_loc: Optional[pathlib.Path] = None,
    save_name: Optional[str] = None,
    save_suffix: str = ".png",
) -> None:
    B = torch.arange(A.shape[0]).numpy()
    A = A.numpy()
    plt.figure(figsize=(width, height))
    ax = sns.barplot(y=A, x=B, palette=sns.color_palette("colorblind"))
    if xlabs is not None:
        ax.set_xticklabels(xlabs, rotation=xrot)
    if ylim is not None:
        plt.ylim(0, ylim)
    if title is not None:
        plt.title(title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if save:
        path = save_loc if save_loc is not None else pathlib.Path.cwd()
        base = save_name if save_name is not None else "efe"
        name = base + save_suffix
        i = 0
        while (path / name).exists():
            name = base + "(" + str(i) + ")" + save_suffix
            i += 1
        plt.savefig(
            path / name, dpi=300, facecolor="w", transparent=False, bbox_inches="tight"
        )
    if show:
        plt.show()
    plt.close()
