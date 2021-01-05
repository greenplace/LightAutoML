"""Tokenizer classes for text preprocesessing and tokenization."""

import re
from functools import partial
from multiprocessing import Pool
from typing import Sequence, Union, List, Optional, Any

import nltk
from log_calls import record_history
from nltk.stem import SnowballStemmer

from .abbreviations import ABBREVIATIONS
from ..dataset.base import RolesDict
from ..dataset.roles import ColumnRole

Roles = Union[Sequence[ColumnRole], ColumnRole, RolesDict, None]


@record_history(enabled=False)
def tokenizer_func(arr, tokenizer):
    """Additional tokenizer function."""
    return [tokenizer._tokenize(x) for x in arr]


@record_history(enabled=False)
class BaseTokenizer:
    """Base class for tokenizer method."""

    _fname_prefix = None
    _fit_checks = ()
    _transform_checks = ()

    def __init__(self, n_jobs: int = 4, to_string: bool = True, use_inverse_index: bool = False, **kwargs: Any):
        """Tokenization with simple text cleaning and preprocessing.

        Args:
            n_jobs: number of threads for multiprocessing.
            to_string: return string or list of tokens.

        """
        super().__init__(**kwargs)
        self.n_jobs = n_jobs
        self.to_string = to_string

        self.stopwords = False
        self.use_inverse_index = use_inverse_index
        self.linedelimeter = ''
        self.worddelimeter = ''
        self.token_linedelimeter = ''
        self.token_worddelimeter = ''

    def _get_delimeters(self, snt: List[str]):
        delim_array = []
        chars = set(''.join(snt))
        if isinstance(self.stopwords, (tuple, list, set)):
            chars = chars | set(''.join(self.stopwords))
        for i in range(1,31):
            if chr(i) not in chars:
                delim_array.append(chr(i))
                if len(delim_array) == 2:
                    return delim_array
        for i in range(20000,30000):
            if chr(i) not in chars:
                delim_array.append(chr(i))
                if len(delim_array) == 2:
                    return delim_array
        return delim_array

    def preprocess_sentence(self, snt: str) -> str:
        """Preprocess sentence string (lowercase, etc.).

        Args:
            snt: sentence string.

        Returns:
            resulting string.

        """
        return snt

    def tokenize_sentence(self, snt: str) -> List[str]:
        """Convert sentence string to a list of tokens.

        Args:
            snt: sentence string.

        Returns:
            resulting list of tokens.

        """
        return snt.split(' ')

    def filter_tokens(self, snt: List[str]) -> List[str]:
        """Clean list of sentence tokens.

        Args:
            snt: list of tokens.

        Returns:
            resulting list of filtered tokens

        """
        return snt

    def postprocess_tokens(self, snt: List[str]) -> List[str]:
        """Additional processing steps: lemmatization, pos tagging, etc.

        Args:
            snt: list of tokens.

        Returns:
            resulting list of processed tokens.
        """
        return snt

    def postprocess_sentence(self, snt: str) -> str:
        """Postprocess sentence string (merge words).

        Args:
            snt: sentence string.

        Returns:
            resulting string.

        """
        return snt

    def _tokenize(self, snt: str) -> Union[List[str], str]:
        """Tokenize text string

        Args:
            snt: string.

        Returns:
            resulting tokenized list.

        """
        res = self.preprocess_sentence(snt)
        res = self.tokenize_sentence(res)
        res = self.filter_tokens(res)
        res = self.postprocess_tokens(res)

        if self.to_string:
            res = ' '.join(res)
            res = self.postprocess_sentence(res)
        return res

    def tokenize(self, text: List[str]) -> Union[List[List[str]], List[str]]:
        """Tokenize list of texts.

        Args:
            text: list of texts.

        Returns:
            resulting tokenized list.

        """

        if self.use_inverse_index:
            # Get 2 unique chars
            delimeters = self._get_delimeters(text)
            self.linedelimeter, self.worddelimeter = delimeters[0], delimeters[1]
            # Get tokenized chars
            self.token_linedelimeter, self.token_worddelimeter = self._tokenize(self.linedelimeter), self._tokenize(self.worddelimeter)
            # Get one array from list
            fulldata = (' '+self.linedelimeter+' ').join(text).split(' ')
            # Create uniq list (u) and inverse index (indices)
            indices = [0 for _ in fulldata]
            counter = {}
            for idx, word in enumerate(fulldata):
                try:
                    indices[idx] = counter[word]
                except:
                    pos = len(counter)
                    counter[word] = pos
                    indices[idx] = pos
            u = ['' for _ in counter.keys()]
            for key in counter.keys():
                u[counter[key]] = key

            del fulldata
            # Split to some jobs if nessesary
            text = [(' '+self.worddelimeter+' ').join(part) for part in self._get_chanks(list(u), self.n_jobs)]

        if self.n_jobs == 1:
            res = self._tokenize_singleproc(text)
        else:
            res = self._tokenize_multiproc(text)

        if self.use_inverse_index:
            data = res
            if self.to_string:
                data = [node.split(self.token_worddelimeter) for node in res]
            elements = []
            [elements.extend(node) for node in data]
            lines = [elements[i] for i in indices]
            res = ' '.join(lines).split(self.token_linedelimeter)
            if not self.to_string:
                res = [node.split(' ') for node in res]

        return res

    def _tokenize_singleproc(self, snt: List[str]) -> List[str]:
        """Singleproc version of tokenization.

        Args:
            snt: list of texts.

        Returns:
            list of tokenized texts.

        """
        return [self._tokenize(x) for x in snt]

    def _get_chanks(self, snt: List[str], n: int):
        idx = list(range(0, len(snt), len(snt) // n + 1)) + [len(snt)]
        parts = [snt[i: j] for (i, j) in zip(idx[:-1], idx[1:])]
        return parts

    def _tokenize_multiproc(self, snt: List[str]) -> List[str]:
        """Multiproc version of tokenization.

        Args:
            snt: list of texts.

        Returns:
            list of tokenized texts.

        """
        parts = self._get_chanks(snt, self.n_jobs)

        f = partial(tokenizer_func, tokenizer=self)
        with Pool(self.n_jobs) as p:
            res = p.map(f, parts)
        del f

        tokens = res[0]
        for r in res[1:]:
            tokens.extend(r)
        return tokens


@record_history(enabled=False)
class SimpleRuTokenizer(BaseTokenizer):
    """Russian tokenizer."""

    def __init__(self, n_jobs: int = 4, to_string: bool = True, use_inverse_index: bool = False, 
                 stopwords: Optional[Union[bool, Sequence[str]]] = False,
                 is_stemmer: bool = True, **kwargs: Any):
        """Tokenizer for Russian language.

        Include numeric, abbreviations, punctuation and short word filtering. Use stemmer by default and do lowercase.

        Args:
            n_jobs: number of threads for multiprocessing.
            to_string: return string or list of tokens.
            stopwords: use stopwords or not.
            is_stemmer: use stemmer.

        """

        super().__init__(n_jobs, **kwargs)
        self.n_jobs = n_jobs
        self.to_string = to_string
        self.use_inverse_index = use_inverse_index
        if isinstance(stopwords, (tuple, list, set)):
            self.stopwords = set(stopwords)
        elif stopwords:
            self.stopwords = set(nltk.corpus.stopwords.words('russian'))
        else:
            self.stopwords = {}

        self.stemmer = SnowballStemmer('russian', ignore_stopwords=len(self.stopwords) > 0) if is_stemmer else None

    @staticmethod
    def _is_abbr(word: str) -> bool:
        """Check if the word is an abbreviation."""

        return sum([x.isupper() and x.isalpha() for x in word]) > 1 and len(word) <= 5

    def preprocess_sentence(self, snt: str) -> str:
        """Preprocess sentence string (lowercase, etc.).

        Args:
            snt: sentence string.

        Returns:
            resulting string.

        """
        snt = snt.strip()
        s = re.sub('[^A-Za-zА-Яа-я0-9' + self.linedelimeter + self.worddelimeter + ']+', ' ', snt)
        s = re.sub(r'^\d+\s|\s\d+\s|\s\d+$', ' ', s)
        return s

    def tokenize_sentence(self, snt: str) -> List[str]:
        """Convert sentence string to a list of tokens.

        Args:
            snt: sentence string.

        Returns:
            resulting list of tokens.

        """
        return snt.split(' ')

    def filter_tokens(self, snt: List[str]) -> List[str]:
        """Clean list of sentence tokens.

        Args:
            snt: list of tokens.

        Returns:
            resulting list of filtered tokens.

        """

        filtered_s = ['' for _ in snt]
        for idx, w in enumerate(snt):

            # ignore numbers
            if w.isdigit():
                pass
            elif w.lower() in self.stopwords:
                pass
            elif w.lower() in ABBREVIATIONS:
                # filtered_s.extend(ABBREVIATIONS[w.lower()].split())
                filtered_s[idx] = ABBREVIATIONS[w.lower()]
            elif self._is_abbr(w) or w in [self.worddelimeter, self.linedelimeter]:
                # filtered_s.append(w)
                filtered_s[idx] = w
            # ignore short words
            elif len(w) < 2:
                pass
            elif w.isalpha():
                # filtered_s.append(w.lower())
                filtered_s[idx] = w.lower()
        return filtered_s

    def postprocess_tokens(self, snt: List[str]) -> List[str]:
        """Additional processing steps: lemmatization, pos tagging, etc.

        Args:
            snt: list of tokens.

        Returns:
            resulting list of processed tokens.

        """
        if self.stemmer is not None:
            return [self.stemmer.stem(w.lower()) for w in snt]
        else:
            return snt

    def postprocess_sentence(self, snt: str) -> str:
        """Postprocess sentence string (merge words).

        Args:
            snt: sentence string.

        Returns:
            resulting string.

        """
        snt = snt.replace('не ', 'не')
        snt = snt.replace('ни ', 'ни')
        return snt


@record_history(enabled=False)
class SimpleEnTokenizer(BaseTokenizer):
    """English tokenizer."""

    def __init__(self, n_jobs: int = 4, to_string: bool = True, use_inverse_index: bool = False,
                 stopwords: Optional[Union[bool, Sequence[str]]] = False,
                 is_stemmer: bool = True, **kwargs: Any):
        """Tokenizer for English language.

        Args:
            n_jobs: number of threads for multiprocessing.
            to_string: return string or list of tokens.
            stopwords: use stopwords or not.
            is_stemmer: use stemmer.

        """

        super().__init__(n_jobs, **kwargs)
        self.n_jobs = n_jobs
        self.to_string = to_string
        self.use_inverse_index = use_inverse_index
        if isinstance(stopwords, (tuple, list, set)):
            self.stopwords = set(stopwords)
        elif stopwords:
            self.stopwords = set(nltk.corpus.stopwords.words('english'))
        else:
            self.stopwords = {}

        self.stemmer = SnowballStemmer('english', ignore_stopwords=len(self.stopwords) > 0) if is_stemmer else None

    def preprocess_sentence(self, snt: str) -> str:
        """Preprocess sentence string (lowercase, etc.).

        Args:
            snt: sentence string.

        Returns:
            resulting string.

        """
        return snt

    def tokenize_sentence(self, snt: str) -> List[str]:
        """Convert sentence string to a list of tokens.

        Args:
            snt: sentence string.

        Returns:
            resulting list of tokens.

        """
        return snt.split(' ')

    def filter_tokens(self, snt: List[str]) -> List[str]:
        """Clean list of sentence tokens.

        Args:
            snt: list of tokens.

        Returns:
            resulting list of filtered tokens.

        """
        if len(self.stopwords) > 0:
            filtered_s = ['' for _ in snt]
            for idx, w in enumerate(snt):
                if w.lower() not in self.stopwords:
                    # filtered_s.append(w)
                    filtered_s[idx] = w
            return filtered_s
        else:
            return snt

    def postprocess_tokens(self, snt: List[str]) -> List[str]:
        """Additional processing steps: lemmatization, pos tagging, etc.

        Args:
            snt: list of tokens.

        Returns:
            resulting list of processed tokens.

        """
        if self.stemmer is not None:
            return [self.stemmer.stem(w.lower()) for w in snt]
        else:
            return snt

    def postprocess_sentence(self, snt: str) -> str:
        """Postprocess sentence string (merge words).

        Args:
            snt: sentence string.

        Returns:
            resulting string.

        """
        return snt
