import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import click
from os import path


@click.group()
def cli():
    pass


@cli.command()
@click.argument("directory")
@click.argument("x")
@click.argument("y")
@click.option("--hue", type=str, default="Method")
@click.option("--save/--no-save", default=True)
@click.option("--show/--no-show", default=True)
@click.option("--filter", type=str, multiple=True, default=[])
@click.option("--facet_x", type=str, default=None)
@click.option("--facet_y", type=str, default=None)
def lineplot(directory, x, y, hue, save, show, filter, facet_x, facet_y):
    observations = pd.read_csv(path.join(directory, "stats.csv"))
    for f in filter:
        observations = eval("observations[" + f + "]")
    if facet_x is not None or facet_y is not None:
        g = sns.FacetGrid(observations, col=facet_x, row=facet_y)
        g = g.map(sns.lineplot, x, y, hue=hue)
    else:
        sns.lineplot(x, y, hue=hue, data=observations)
    plt.xlabel(x)
    plt.ylabel(y)
    if save:
        X = x.replace(" ", "_")
        Y = y.replace(" ", "_")
        plt.savefig(path.join(directory, "{}-{}.png".format(X, Y)), dpi=320)
    if show:
        plt.show()

if __name__ == "__main__":
    cli()