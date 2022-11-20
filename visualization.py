import matplotlib.pyplot as plt
import numpy as np

import seaborn as sns
import pandas as pd
from gensim.models import Word2Vec
from sklearn.manifold import TSNE


def tsne_scatter_plot(model: Word2Vec, word: str):
    """Plot in seaborn the results from the t-SNE dimensionality
    reduction algorithm of the vectors of a query word, its list
    of most similar words, and a list of words.

    Source: https://github.com/drelhaj/NLP_ML_Visualization_Tutorial
    I did a little bit of refactoring though.
    """
    vectors, labels, colors = get_plot_data(model, word)
    embedded_words = TSNE(n_components=2,
                          random_state=0,
                          perplexity=6,
                          init='pca').fit_transform(np.array(vectors))
    df = pd.DataFrame({'x': embedded_words[:, 0],
                       'y': embedded_words[:, 1],
                       'words': labels,
                       'color': colors})
    plot_tsne(model, df)


def get_plot_data(model: Word2Vec,
                  word: str) -> tuple[list[np.ndarray], list[str], list[str]]:
    """Find most similar words and returns their vectors,
    labels and colors, including origin word.
    """
    vectors = [model.wv.get_vector(word)]
    labels = [word]
    colors = ['red']
    similar_words = model.wv.most_similar(word)
    for key, _ in similar_words:
        wrd_vector = model.wv.get_vector(key)
        vectors.append(wrd_vector)
        labels.append(key)
        colors.append('blue')
    return vectors, labels, colors


def plot_tsne(model: Word2Vec, df: pd.DataFrame):
    """Plot reduced results."""
    plt.figure(figsize=(9, 9))
    plt.title(f'vector_size={model.vector_size}, sg={model.sg}')
    sns.regplot(data=df, x='x', y='y', fit_reg=False, marker='o',
                scatter_kws={'s': 40, 'facecolors': df['color']})
    add_plot_annotations(df)
    crop_plot(df)
    plt.show()


def add_plot_annotations(df: pd.DataFrame):
    """Annotate each dot on the plot."""
    for i in range(df.shape[0]):
        plt.text(df['x'][i],
                 df['y'][i],
                 ' ' + df['words'][i].title(),
                 horizontalalignment='left',
                 verticalalignment='bottom',
                 size='medium',
                 color=df['color'][i],
                 weight='normal').set_size(15)


def crop_plot(df: pd.DataFrame):
    """Crop plot axes."""
    plt.xlim(df['x'].min() - 50, df['x'].max() + 50)
    plt.ylim(df['y'].min() - 50, df['y'].max() + 50)
