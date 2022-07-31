import torch
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from util import util_tokenizer


def visualize_PCA(title, data, data_index, data_label, class_num, output_dir='../figures/'):
    plt.rcParams['savefig.dpi'] = 300  # 图片像素
    plt.rcParams['figure.dpi'] = 300  # 分辨率

    X_pca = PCA(n_components=2).fit_transform(data)
    font = {"color": "darkred", "size": 13, "family": "serif"}
    # plt.style.use("dark_background")
    plt.style.use("default")
    plt.figure()

    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=data_index, alpha=0.6, cmap=plt.cm.get_cmap('rainbow', class_num))
    if data_label:
        for i in range(len(X_pca)):
            plt.annotate(data_label[i], xy=(X_pca[:, 0][i], X_pca[:, 1][i]),
                         xytext=(X_pca[:, 0][i] + 0.00, X_pca[:, 1][i] + 0.00))
    plt.title(title, fontdict=font)

    if data_label is None:
        cbar = plt.colorbar(ticks=range(class_num))
        # cbar.set_label(label='digit value', fontdict=font)
        plt.clim(0 - 0.5, class_num - 0.5)
    plt.savefig(output_dir + '{}.pdf'.format(title))
    plt.show()


def visualize_t_SNE(title, data, data_index, data_label, class_num, output_dir='../figures/'):
    plt.rcParams['savefig.dpi'] = 300  # 图片像素
    plt.rcParams['figure.dpi'] = 300  # 分辨率

    X_tsne = TSNE(n_components=2).fit_transform(data)  # [num_samples, n_components]
    font = {"color": "darkred", "size": 13, "family": "serif"}
    # plt.style.use("dark_background")
    plt.style.use("default")

    plt.figure()
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=data_index, alpha=0.6, cmap=plt.cm.get_cmap('rainbow', class_num))
    if data_label:
        for i in range(len(X_tsne)):
            plt.annotate(data_label[i], xy=(X_tsne[:, 0][i], X_tsne[:, 1][i]),
                         xytext=(X_tsne[:, 0][i] + 1, X_tsne[:, 1][i] + 1))
    plt.title(title, fontdict=font)

    if data_label is None:
        cbar = plt.colorbar(ticks=range(class_num))
        # cbar.set_label(label='digit value', fontdict=font)
        plt.clim(0 - 0.5, class_num - 0.5)
    plt.savefig(output_dir + '{}.pdf'.format(title))
    plt.show()


def draw_residue_heatmap(visual_std_acid_tensor, xticklabels, yticklabels, title, annot=True, output_dir='../figures/'):
    plt.rcParams['savefig.dpi'] = 300  # 图片像素
    plt.rcParams['figure.dpi'] = 300  # 分辨率

    f, ax = plt.subplots(figsize=(14, 4))
    plt.subplots_adjust(top=0.85, wspace=0.2, hspace=0.3)
    ax = sns.heatmap(visual_std_acid_tensor, xticklabels=xticklabels, yticklabels=yticklabels, linewidths=0.5,
                     cmap="YlGnBu",
                     annot=annot)
    ax.set_title(title, fontsize=18)
    plt.savefig(output_dir + '{}.pdf'.format(title))
    plt.show()


def draw_position_heatmap(visual_position_tensor, xticklabels, yticklabels, title, annot=False,
                          output_dir='../figures/'):
    plt.rcParams['savefig.dpi'] = 300  # 图片像素
    plt.rcParams['figure.dpi'] = 300  # 分辨率

    f, ax = plt.subplots(figsize=(40, 4))
    plt.subplots_adjust(top=0.85, wspace=0.2, hspace=0.3)
    ax = sns.heatmap(visual_position_tensor, xticklabels=xticklabels, yticklabels=yticklabels, linewidths=0.5,
                     cmap="YlGnBu",
                     annot=annot)
    ax.set_title(title, fontsize=18)
    plt.savefig(output_dir + '{}.pdf'.format(title))
    plt.show()
