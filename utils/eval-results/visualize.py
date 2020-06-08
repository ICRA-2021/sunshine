#!/usr/bin/env python3
import csv
import click
from os import path
import pandas as pd
import seaborn as sns
from scipy.stats import lognorm
import numpy as np
import matplotlib.pyplot as plt


def load_observations(dir):
    samples_fname = path.join(dir, "samples.dat")
    observations_fname = path.join(dir, "aggregated_observations.dat")
    observation_dicts = []
    with open(samples_fname, "r") as samples_file:
        samples_file.readline()
        for line in samples_file:
            i, a, b, g, s = line.split()
            alpha = lognorm.ppf(float(a), 2.25, scale=np.exp(-2.25))
            beta = lognorm.ppf(float(b), 2.0, scale=np.exp(-2.5))
            gamma = np.power(10, 3.5 * np.log(float(g)))
            cell_space = lognorm.ppf(float(s), 0.6, scale=np.exp(-0.7))
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
def heatmap(directory, x, y):
    observations = load_observations(directory)
    xscale = 'log' #if x in ['Alpha', 'Beta', 'Gamma'] else 'linear'
    yscale = 'log' #if y in ['Alpha', 'Beta', 'Gamma'] else 'linear'
    plt.hexbin(observations[x], observations[y], C=observations["Score"], gridsize=20, cmap="copper", xscale=xscale, yscale=yscale)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    cli()
