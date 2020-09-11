import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

sns.set_style("dark")
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 12

with open('f0_dist.pkl', 'rb') as f:
    logf0s = pickle.load(f)

models = ['GT', 'StarGAN-VC', 'AutoVC', 'Seq2seq-VC', 'DCVC']
cases = ['m2f', 'f2m']

plot_data = np.array([])
plot_labels = []
plot_case = []
for model in models:
    for c in cases:
        data = logf0s[model][c]
        plot_data = np.concatenate([plot_data, data])
        plot_labels += [model] * data.shape[0]
        plot_case += [c] * data.shape[0]

plot_dict = {'log(f0)': plot_data,
             'label': plot_labels,
             'case': plot_case}
df = pd.DataFrame(plot_dict)

#figure, ((ax1, ax3, ax5, ax7), (ax2, ax4, ax6, ax7)) = plt.subplots(nrows=4, ncols=2)
fig, ax = plt.subplots(nrows=4, ncols=2, figsize=(7, 9))

for idx1, m in enumerate(['StarGAN-VC', 'AutoVC', 'Seq2seq-VC', 'DCVC']):
    for idx2, c in enumerate(['m2f', 'f2m']):
        sns.distplot(df[(df.label == m) & (df.case == c)]["log(f0)"], color="red", hist=False, label=m,
                     ax=ax[idx1][idx2])
        sns.distplot(df[(df.label == "GT") & (df.case == c)]["log(f0)"], color="blue", hist=False, label="GT", ax=ax[idx1][idx2])

        ax[idx1][idx2].set_xlim(4, 6)
        ax[idx1][idx2].set_ylim(0, 3.5)
        ax[idx1][idx2].set_title('M2F' if c == 'm2f' else 'F2M')
        ax[idx1][idx2].set_yticks([])

        if idx2 == 0:
            ax[idx1][idx2].set_ylabel("Probability")
            plt.legend(loc='upper left')
        else:
            plt.legend(loc='upper right')

        # Get the two lines from the axes to generate shading
        l1 = ax[idx1][idx2].lines[0]
        l2 = ax[idx1][idx2].lines[1]

        # Get the xy data from the lines so that we can shade
        x1 = l1.get_xydata()[:, 0]
        y1 = l1.get_xydata()[:, 1]
        x2 = l2.get_xydata()[:, 0]
        y2 = l2.get_xydata()[:, 1]
        ax[idx1][idx2].fill_between(x1, y1, color="red", alpha=0.3)
        ax[idx1][idx2].fill_between(x2, y2, color="blue", alpha=0.3)

plt.show()
fig.savefig('f0dist.eps', dpi=1000, format='eps')
fig.savefig('f0dist.png', dpi=1000, format='png')