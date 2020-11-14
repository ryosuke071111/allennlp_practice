import os
import tempfile
import logging
import sentencepiece as spm
from typing import List
from overrides import overrides

from allennlp.common import Params
from allennlp.data.tokenizers.token import Token
from allennlp.data.tokenizers.tokenizer import Tokenizer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Tokenizer.register("sentencepiece")
class SentencePieceTokenizer(Tokenizer):
    """
    A ``WordTokenizer`` handles the splitting of strings into words as well as any desired
    post-processing (e.g., stemming, filtering, etc.).  Note that we leave one particular piece of
    post-processing for later: the decision of whether or not to lowercase the token.  This is for
    two reasons: (1) if you want to make two different casing decisions for whatever reason, you
    won't have to run the tokenizer twice, and more importantly (2) if you want to lowercase words
    for your word embedding, but retain capitalization in a character-level representation, we need
    to retain the capitalization here.
    Parameters
    ----------
    word_splitter : ``WordSplitter``, optional
        The :class:`WordSplitter` to use for splitting text strings into word tokens.  The default
        is to use the ``JustSpacesWordSplitter``, which is non destructive other than for spaces.
    word_filter : ``WordFilter``, optional
        The :class:`WordFilter` to use for, e.g., removing stopwords.  Default is to do no
        filtering.
    word_stemmer : ``WordStemmer``, optional
        The :class:`WordStemmer` to use.  Default is no stemming.
    start_tokens : ``List[str]``, optional
        If given, these tokens will be added to the beginning of every string we tokenize.
    end_tokens : ``List[str]``, optional
        If given, these tokens will be added to the end of every string we tokenize.
    """

    def __init__(self,
                 model_path: str,
                 vocab_size: int = 8000,
                 model_type: str = 'unigram') -> None:
        # We reverse the tokens here because we're going to insert them with `insert(0)` later;
        # this makes sure they show up in the right order.
        self._vocab_size = vocab_size
        self._model_path = model_path
        self._model = spm.SentencePieceProcessor()
        self._model_type = model_type
        self.trained = False
        if os.path.exists(self._model_path):
            self._model.Load(self._model_path)
            self.trained = True


    @overrides
    def batch_tokenize(self, texts: List[str]) -> List[List[Token]]:
        preprocessed_texts = self._batch_preprocess(texts)
        return [self._tokenize(text) for text in preprocessed_texts]


    @overrides
    def tokenize(self, text: str) -> List[Token]:
        str_tokens = self._model.EncodeAsPieces(text)
        tokens = [Token(token, i) for i, token in enumerate(str_tokens)]
        return tokens
