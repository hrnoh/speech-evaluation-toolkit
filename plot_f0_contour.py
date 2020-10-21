import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

sns.set_style("dark")
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 7

# 10번 좋음
idx = 14
start = 10
end = 50

with open('f0_contour.pkl', 'rb') as f:
    logf0s = pickle.load(f)

models = ['GT', 'StarGAN-VC', 'AutoVC', 'Seq2seq-VC', 'DCVC']
cases = ['m2f', 'f2m']

# plot_data = np.array([])
# plot_labels = []
# plot_case = []
# for model in models:
#     for c in cases:
#         data = logf0s[model][c]
#         plot_data = np.concatenate([plot_data, data])
#         plot_labels += [model] * data.shape[0]
#         plot_case += [c] * data.shape[0]

# plot_dict = {'log(f0)': plot_data,
#              'label': plot_labels,
#              'case': plot_case}
# df = pd.DataFrame(plot_dict)

#figure, ((ax1, ax3, ax5, ax7), (ax2, ax4, ax6, ax7)) = plt.subplots(nrows=4, ncols=2)
fig, ax = plt.subplots(nrows=4, ncols=2, figsize=(7, 9))

for idx1, m in enumerate(['StarGAN-VC', 'AutoVC', 'Seq2seq-VC', 'DCVC']):
    for idx2, c in enumerate(['m2f', 'f2m']):
        # sns.distplot(df[(df.label == m) & (df.case == c)]["log(f0)"], color="red", hist=False, label=m,
        #              ax=ax[idx1][idx2])
        # sns.distplot(df[(df.label == "GT") & (df.case == c)]["log(f0)"], color="blue", hist=False, label="GT", ax=ax[idx1][idx2])
        sns.lineplot(data=logf0s[c][idx][m], color='red', label=m, linewidth=1, ax=ax[idx1][idx2])
        sns.lineplot(data=logf0s[c][idx]['GT'], color='blue', label='GT', linewidth=1, ax=ax[idx1][idx2])

        #ax[idx1][idx2].set_xlim(4, 6)
        #ax[idx1][idx2].set_ylim(0, 3.5)
        ax[idx1][idx2].set_title('M2F' if c == 'm2f' else 'F2M')
        ax[idx1][idx2].set_yticks([])

        if idx2 == 0:
            ax[idx1][idx2].set_ylabel("log(f0)")
            # plt.legend(loc='upper left')
            #ax[idx1][idx2].legend(loc='center right', bbox_to_anchor=(1.25, 0.5), ncol=1)
            ax[idx1][idx2].legend(loc='upper left')
        else:
            ax[idx1][idx2].legend(loc='upper right')
            # plt.legend(loc='upper right')

plt.tight_layout(pad=3)
plt.show()
# fig.savefig('f0contour.eps', dpi=1000, format='eps')
# fig.savefig('f0contour.png', dpi=1000, format='png')