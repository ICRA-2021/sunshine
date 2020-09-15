#!/usr/bin/env python3
import csv
import click
from os import path
import pandas as pd
import seaborn as sns
from scipy.stats import lognorm
import numpy as np
import matplotlib.pyplot as plt

CUR_VERSION = 6

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
    if version >= 4:
        return pow(3, float(x)) / 2.
    if version >= 2:
        return lognorm.ppf(float(x), 1.0, scale=np.exp(0.0))
    if version >= 1:
        return lognorm.ppf(float(x), 0.6, scale=np.exp(-0.7))
    raise ValueError("Unrecognized version number.")

def compute_clahe(x, version):
    if version >= 3:
        return int(float(x) >= 0.5)
    raise ValueError("Variable not defined in this version.")


def compute_texton(x, version):
    if version >= 3:
        return int(float(x) >= 0.75)
    raise ValueError("Variable not defined in this version.")


def compute_orb(x, version):
    if version >= 3:
        return int(float(x) >= 0.5)
    raise ValueError("Variable not defined in this version.")


def parse_sample(line, version):
    tokens = line.split()
    sample = {}
    if version >= 1:
        i, a, b, g = tokens[:4]
        sample["Iteration"] = int(i)
        sample["Alpha"] = compute_alpha(float(a), version)
        sample["Beta"] = compute_beta(float(b), version)
    if version < 6:
        sample["Gamma"] = compute_gamma(float(g), version)
    if version < 5:
        s = tokens[4]
        sample["Cell Space"] = compute_cell_space(float(s), version)
    if version >= 3:
        clahe = tokens[5 if version < 5 else 4 if version == 5 else 3]
        sample["CLAHE"] = compute_clahe(clahe, version)
    if version == 3:
        texton, orb = tokens[6:8]
        sample["Texton"] = compute_texton(texton, version)
        sample["ORB"] = compute_orb(orb, version)
    return sample


def load_observations(dir, version):
    samples_fname = path.join(dir, "samples.dat")
    observations_fname = path.join(dir, "aggregated_observations.dat")
    observation_dicts = []
    with open(samples_fname, "r") as samples_file:
        samples_file.readline()
        for line in samples_file:
            observation_dicts.append(parse_sample(line, version))
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


def hexbin(x, y, z, color, **kwargs):
    cmap = sns.light_palette(color, as_cmap=True)
    plt.hexbin(x, y, C=z, **kwargs)

@cli.command()
@click.argument("directory")
@click.argument("x")
@click.argument("y")
@click.option("--version", type=int, default=CUR_VERSION)
@click.option("--save/--no-save", default=True)
@click.option("--show/--no-show", default=True)
@click.option("--filter", type=str, multiple=True, default=[])
@click.option("--facet_x", type=str, default=None)
@click.option("--facet_y", type=str, default=None)
def heatmap(directory, x, y, version, save, show, filter, facet_x, facet_y):
    observations = load_observations(directory, version=version)
    observations.to_csv(path.join(directory, "processed_observations-v{}.csv".format(version)))
    for f in filter:
        observations = eval("observations[" + f + "]")
    xscale = 'log' #if x in ['Alpha', 'Beta', 'Gamma'] else 'linear'
    yscale = 'log' #if y in ['Alpha', 'Beta', 'Gamma'] else 'linear'
    if facet_x is not None or facet_y is not None:
        g = sns.FacetGrid(observations, col=facet_x, row=facet_y)
        g = g.map(hexbin, x, y, "Score", gridsize=20, cmap="copper", xscale=xscale, yscale=yscale)
    else:
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
