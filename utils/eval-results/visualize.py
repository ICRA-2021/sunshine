#!/usr/bin/env python3
import csv
import click
from os import path
import pandas as pd
import seaborn as sns
from scipy.stats import lognorm
import numpy as np
import matplotlib.pyplot as plt

CUR_VERSION = 2


def compute_alpha(x, version):
    if version >= 1:
        return lognorm.ppf(float(x), 2.25, scale=np.exp(-2.25))
    raise ValueError("Unrecognized version number.")


def compute_beta(x, version):
    if version >= 1:
        return lognorm.ppf(float(x), 2.0, scale=np.exp(-2.5))
    raise ValueError("Unrecognized version number.")


def compute_gamma(x, version):
    if version >= 2:
        return lognorm.ppf(float(x), 4.0, scale=np.exp(-8.0))
    elif version >= 1:
        return np.power(10, 3.5 * np.log(float(x)))
    raise ValueError("Unrecognized version number.")


def compute_cell_space(x, version):
    if version >= 2:
        return lognorm.ppf(float(x), 1.0, scale=np.exp(0.0))
    if version >= 1:
        return lognorm.ppf(float(x), 0.6, scale=np.exp(-0.7))
    raise ValueError("Unrecognized version number.")


def load_observations(dir, version):
    samples_fname = path.join(dir, "samples.dat")
    observations_fname = path.join(dir, "aggregated_observations.dat")
    observation_dicts = []
    with open(samples_fname, "r") as samples_file:
        samples_file.readline()
        for line in samples_file:
            i, a, b, g, s = line.split()
            alpha = compute_alpha(float(a), version)
            beta = compute_beta(float(b), version)
            gamma = compute_gamma(float(g), version)
            cell_space = compute_cell_space(float(s), version)
            # print(alpha, beta, gamma, cell_space)
            observation_dicts.append({"Iteration": int(i), "Alpha": alpha, "Beta": beta, "Gamma": gamma, "Cell Space": cell_space})
    with open(observations_fname, "r") as observations_file:
        observations_file.readline()
        for line, observation_dict in zip(observations_file, observation_dicts):
            i, score = line.split()
            idx = int(i)
            assert idx == observation_dict["Iteration"]
            observation_dict["Score"] = float(score)
    return pd.DataFrame(observation_dicts)

@click.group()
def cli():
    pass


@cli.command()
@click.argument("directory")
@click.argument("x")
@click.argument("y")
@click.option("--version", type=int, default=CUR_VERSION)
@click.option("--save/--no-save", default=True)
@click.option("--show/--no-show", default=True)
def heatmap(directory, x, y, version, save, show):
    observations = load_observations(directory, version=version)
    observations.to_csv(path.join(directory, "processed_observations-v{}.csv".format(version)))
    xscale = 'log' #if x in ['Alpha', 'Beta', 'Gamma'] else 'linear'
    yscale = 'log' #if y in ['Alpha', 'Beta', 'Gamma'] else 'linear'
    plt.hexbin(observations[x], observations[y], C=observations["Score"], gridsize=20, cmap="copper", xscale=xscale, yscale=yscale)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.colorbar()
    if save:
        X = x.replace(" ", "_")
        Y = y.replace(" ", "_")
        plt.savefig(path.join(directory, "{}-{}-v{}.png".format(X, Y, version)), dpi=320)
    if show:
        plt.show()


if __name__ == "__main__":
    cli()
