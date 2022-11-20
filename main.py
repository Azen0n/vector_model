import os
import re
import warnings

import fitz
from gensim.models import Word2Vec
from nltk.stem.snowball import SnowballStemmer

from datatypes import Article
from visualization import tsne_scatter_plot

warnings.simplefilter(action='ignore', category=FutureWarning)
stemmer = SnowballStemmer('russian')


def main():
    models = load_models(directory='models')
    start_ui(models)


def start_ui(models: list[Word2Vec]) -> None:
    """Simple console interface to select model and search through."""
    if len(models) == 0:
        print('No models passed.')
        return
    print(f'Select model:')
    for i, model in enumerate(models):
        print(f'"{i}" - vector_size={model.vector_size}, sg={model.sg}')
    print(f'"{len(models)}" - all of them')
    model_index = input()
    try:
        index = int(model_index)
        if index == len(models):
            search_word(models)
        search_word([models[index]])
    except (TypeError, IndexError, ValueError) as e:
        print('Incorrect option. No second chance.')


def search_word(models: list[Word2Vec]):
    """Start searching loop."""
    word = input('Enter word ("!exit" to exit): ')
    while word != '!exit':
        plot_word(word, models)
        word = input('Enter word: ')


def plot_word(word: str, models: list[Word2Vec]):
    """Plot word if such exist in model."""
    try:
        for model in models:
            tsne_scatter_plot(model, word.lower())
    except KeyError:
        print(f'Word {word} not found.')


def save_models(directory: str = 'models'):
    """Save model objects with sets of parameters to files."""
    text = read_pdf('file.pdf')
    articles = split_text_into_articles(text)
    sentences = [sentence.words for article in articles for sentence in article.sentences]

    params = [{'vector_size': 100, 'min_count': 2, 'window': 10, 'sg': 0},
              {'vector_size': 100, 'min_count': 2, 'window': 10, 'sg': 1},
              {'vector_size': 50, 'min_count': 2, 'window': 10, 'sg': 0}]

    if not os.path.exists(directory):
        os.makedirs(directory)
    for param in params:
        model = Word2Vec(sentences, **param)
        model.save(f'{directory}/vector_size_{param["vector_size"]}_sg_{param["sg"]}.model')


def load_models(directory: str = 'models') -> list[Word2Vec]:
    """Return list of saved model objects."""
    model_names = os.listdir(directory)
    return [Word2Vec.load(f'{directory}/{name}') for name in model_names]


def read_pdf(path: str, save_txt_path: str = None) -> str:
    """Read pdf with scholar articles and returns plain string.

    If save_txt_path passed, text is saved in said path.
    """
    with fitz.open(path) as document:
        text = ''
        for page in document:
            text += page.get_text()
    if save_txt_path:
        with open(save_txt_path, 'w', encoding='utf8') as f:
            f.write(text)
    return text


def split_text_into_articles(text: str) -> list[Article]:
    """Return list of Article objects, consisting of
    list of authors, title and main text without 'СПИСОК ЛИТЕРАТУРЫ'.
    """
    pattern = re.compile(r"""(?:\d+\s?)\n   # page number (ignored)
                         ((?:.+\n){1,5})    # authors and their universities
                         (?:УДК:?\s+[\d]+\.?[\d]+.+\n)  # УДК (ignored)
                         ((?:.+\n){1,5}(?=Аннотация))   # article title
                         (?:Аннотация\.?:?)    # counts as first sentence
                                                # so ignored
                         """, re.X)
    authors_pattern = r'[А-ЯA-Z]\. ?[А-ЯA-Z]\.? [А-Яа-я]+'
    splitted_text = re.split(pattern, text)
    articles = []
    try:
        for i in range(1, len(splitted_text), 3):
            authors = re.findall(authors_pattern, splitted_text[i])
            title = splitted_text[i + 1]
            raw_text = re.split('СПИСОК ЛИТЕРАТУРЫ', splitted_text[i + 2])[0]
            articles.append(Article(authors, title, raw_text))
    except IndexError:
        raise IndexError('Wrong split occurred. Adjust regex pattern.')
    return articles


if __name__ == '__main__':
    # save_models(directory='models')
    main()
