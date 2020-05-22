from argparse import ArgumentParser
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def main(args):
  sns.set_style('darkgrid')

  df = pd.read_table(args.file, header=None)

  fig, axs = plt.subplots(3)
  fig.suptitle(args.file)

  adj_type_order = df[2].value_counts().index

  # Plot adjective type histogram.
  sns.countplot(x=2, order=adj_type_order, data=df, ax=axs[0])

  # Plot type/token ratios.
  token_frequencies = df[2].value_counts()
  type_frequencies = df.groupby(2).agg({0: 'nunique'}).reindex(token_frequencies.index)
  type_token_ratios = type_frequencies.copy().div(token_frequencies.apply(np.log), axis=0)
  type_token_ratios = type_token_ratios.rename({0: "type_token_ratio"}, axis=1).reset_index()

  sns.barplot(x="index", y="type_token_ratio", data=type_token_ratios, ax=axs[1])

  # Type frequencies
  sns.barplot(x="index", y=0, data=type_frequencies.reset_index(), ax=axs[2])

  axs[0].set_title("token frequency")
  axs[0].set_xlabel("adj type")
  axs[1].set_ylabel("token frequency")

  axs[1].set_title("type/token ratio")
  axs[1].set_xlabel("adj type")
  axs[1].set_ylabel("type/token ratio")

  plt.tight_layout(rect=[0, 0.03, 1, 0.95])

  if args.out:
    plt.savefig(args.out)
  else:
    plt.show()

  ret = type_token_ratios.copy()

  # Prep corpus statistics.
  ret["corpus"] = args.file
  ret = ret.assign(token_frequency=token_frequencies.values)
  # ret["token_rel_frequency"] = ret["token_frequency"] / ret["token_frequency"].sum()

  # DEV: retain only shape/color.
  ret = ret[ret["index"].isin(["shape", "color"])]
  ret["token_rel_frequency"] = ret["token_frequency"] / ret["token_frequency"].sum()
  n = ret["token_frequency"].sum()
  ret = ret.assign(stderr=np.sqrt(ret["token_rel_frequency"] * (1 - ret["token_rel_frequency"])) / np.sqrt(n))

  ret.to_csv(sys.stdout, sep="\t")


if __name__ == '__main__':
  p = ArgumentParser()

  p.add_argument("file")
  p.add_argument("--out")

  main(p.parse_args())
