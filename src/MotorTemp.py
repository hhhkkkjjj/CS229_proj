import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns


class MLProj:
    def __init__(self):
        self.dataset = pd.read_csv("pmsm_temperature_data.csv")

    def ProfilePlot(self, filename="plots/ProfileID.png", plot=True):
        #Prifile ID#
        exp_time_count = self.dataset.profile_id.value_counts().sort_values()
        fig = plt.figure(figsize=(17, 12))
        sns.barplot(y=exp_time_count.values, x=exp_time_count.index,
                    order=exp_time_count.index, orient="v")
        plt.title("Sample counts for different profiles", fontsize=16)
        plt.ylabel("Sample count", fontsize=14)
        plt.xlabel("Profile ID", fontsize=14)
        fig.savefig(filename)
        if plot:
            plt.show()

    def CorrPlot(self, filename="plots/CorrPlot.png", plot=True):
        #Correlation Map#
        corrmat = self.dataset.corr()
        fig = plt.figure(figsize=(12, 12))
        cmap = sns.diverging_palette(240, 10, as_cmap=True)
        sns.heatmap(corrmat, annot=True, linewidths=.5, fmt='.2f', mask=np.zeros_like(
            corrmat, dtype=np.bool), cmap=cmap, square=True)
        fig.savefig(filename)
        if plot:
            plt.show()


if __name__ == "__main__":
    P = MLProj()
    P.ProfilePlot(plot=True)
    P.CorrPlot(plot=True)
