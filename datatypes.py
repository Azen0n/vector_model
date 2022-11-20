from __future__ import annotations
from dataclasses import dataclass, field
import re

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from pymorphy2 import MorphAnalyzer

STOPWORDS = set(stopwords.words('russian')
                ).union(set(stopwords.words('english')))
stemmer = SnowballStemmer('russian')
morph = MorphAnalyzer()


@dataclass
class Sentence:
    """Sentence.

    * article — Article object with this sentence.
    * text — preprocessed sentence string.
    * words — list of words in this sentence.
    """
    article: Article
    text: str
    words: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.__sentence_preprocessing()
        self.__split_sentence_into_words()

    def __hash__(self) -> int:
        return hash((self.article, self.text, tuple(self.words)))

    def __sentence_preprocessing(self):
        """Remove special characters and extra whitespaces from sentence."""
        self.text = re.sub(r'[^a-zA-Zа-яА-Я\s-]', '', self.text)
        self.text = re.sub(r'\s+', ' ', self.text)
        self.text = self.text.strip()

    def __split_sentence_into_words(self):
        """Split sentence into list of words."""
        words = word_tokenize(self.text)
        self.words = [morph.normal_forms(word)[0] for word in words if word not in STOPWORDS]


@dataclass
class Article:
    """Article.

    * authors — list of author names of article in format 'И.О. Фамилия'
    * title — article title (ALL CAPS).
    * text — preprocessed article text.
    * sentences — list of sentences in this article.
    """
    authors: list[str]
    title: str
    text: str
    sentences: list[Sentence] = field(default_factory=list)

    def __post_init__(self):
        self.__flatten_article_title()
        self.__text_preprocessing()
        self.__split_into_sentences()

    def __hash__(self) -> int:
        return hash((self.title, tuple(self.authors)))

    def __flatten_article_title(self):
        """Remove newline characters and repeating whitespaces from
        article titles.
        """
        self.title = re.sub(r'\n', '', self.title)
        self.title = re.sub(r'\s{2,}', ' ', self.title)
        self.title = self.title.strip()

    def __text_preprocessing(self):
        """Remove newline characters and 'рис. #' references."""
        self.text = self.text.lower()
        self.text = re.sub('\n', '', self.text)
        self.text = re.sub(r'рис\.\s?\d\.?', '', self.text)

    def __split_into_sentences(self):
        """Split article text into sentences."""
        sentences = sent_tokenize(self.text)
        for sentence in sentences:
            self.sentences.append(Sentence(self, sentence))

    def __log(self):
        print(f'Article completed: {self.title}')
        print(f'Number of sentences: {len(self.sentences)}')
        word_count = 0
        for sentence in self.sentences:
            word_count += len(sentence.words)
        print(f'Number of words after preprocessing: {word_count}\n')
