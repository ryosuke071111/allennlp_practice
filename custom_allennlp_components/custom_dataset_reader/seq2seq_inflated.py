import csv
from typing import Dict, Optional
import logging
import copy

from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.dataset_readers.dataset_reader import DatasetReader, AllennlpDataset
from allennlp.data.fields import TextField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, SpacyTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers.token import Token

logger = logging.getLogger(__name__)

import random

num_virtual_models = 9

@DatasetReader.register("seq2seq_inflated")
class PseudoSeq2SeqDatasetReader(DatasetReader):
    """
    Read a tsv file containing paired sequences, and create a dataset suitable for a
    `ComposedSeq2Seq` model, or any model with a matching API.

    Expected format for each input line: <source_sequence_string>\t<target_sequence_string>

    The output of `read` is a list of `Instance` s with the fields:
        source_tokens : `TextField` and
        target_tokens : `TextField`

    `START_SYMBOL` and `END_SYMBOL` tokens are added to the source and target sequences.

    # Parameters

    source_tokenizer : `Tokenizer`, optional
        Tokenizer to use to split the input sequences into words or other kinds of tokens. Defaults
        to `SpacyTokenizer()`.
    target_tokenizer : `Tokenizer`, optional
        Tokenizer to use to split the output sequences (during training) into words or other kinds
        of tokens. Defaults to `source_tokenizer`.
    source_token_indexers : `Dict[str, TokenIndexer]`, optional
        Indexers used to define input (source side) token representations. Defaults to
        `{"tokens": SingleIdTokenIndexer()}`.
    target_token_indexers : `Dict[str, TokenIndexer]`, optional
        Indexers used to define output (target side) token representations. Defaults to
        `source_token_indexers`.
    source_add_start_token : `bool`, (optional, default=`True`)
        Whether or not to add `START_SYMBOL` to the beginning of the source sequence.
    source_add_end_token : `bool`, (optional, default=`True`)
        Whether or not to add `END_SYMBOL` to the end of the source sequence.
    delimiter : `str`, (optional, default=`"\t"`)
        Set delimiter for tsv/csv file.
    quoting : `int`, (optional, default=`csv.QUOTE_MINIMAL`)
        Quoting to use for csv reader.
    """

    def __init__(
        self,
        source_tokenizer: Tokenizer = None,
        target_tokenizer: Tokenizer = None,
        source_token_indexers: Dict[str, TokenIndexer] = None,
        target_token_indexers: Dict[str, TokenIndexer] = None,
        source_add_start_token: bool = True,
        source_add_end_token: bool = True,
        target_add_start_token: bool = True,
        target_add_end_token: bool = True,
        start_symbol: str = START_SYMBOL,
        end_symbol: str = END_SYMBOL,
        delimiter: str = "\t",
        source_max_tokens: Optional[int] = None,
        target_max_tokens: Optional[int] = None,
        quoting: int = csv.QUOTE_MINIMAL,
        pseudo: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._source_tokenizer = source_tokenizer or SpacyTokenizer()
        self._target_tokenizer = target_tokenizer or self._source_tokenizer
        self._source_token_indexers = source_token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._target_token_indexers = target_token_indexers or self._source_token_indexers
        self._source_add_start_token = source_add_start_token
        self._source_add_end_token = source_add_end_token
        self._target_add_start_token = target_add_start_token
        self._target_add_end_token = target_add_end_token
        self._start_token = "@start@"
        self._end_token = "@end@"
        # self._source_tokenizer.tokenize(
        #     start_symbol + " " + end_symbol
        # )
        self._70 = 0
        self._delimiter = delimiter
        self._source_max_tokens = source_max_tokens
        self._target_max_tokens = target_max_tokens
        self._source_max_exceeded = 0
        self._target_max_exceeded = 0
        self.quoting = quoting
        self.pseudo = pseudo
        self.tags = ["[pseudo1]","[pseudo2]","[pseudo3]","[pseudo4]","[pseudo5]", "[pseudo6]","[pseudo7]","[pseudo8]","[pseudo9]"][:num_virtual_models]
        self.s_dic = {}
        self.t_dic = {}
        

    @overrides
    def _read(self, file_path: str, bagging = False):
        if self.pseudo:
            print("Now I added the pseudo tokens to the beginnning of soure!!!! not target!!!!!!")

        pseudo_tags = ["[pseudo1]","[pseudo2]","[pseudo3]","[pseudo4]","[pseudo5]", "[pseudo6]","[pseudo7]","[pseudo8]","[pseudo9]"][:num_virtual_models]

        ret = []

        # Reset exceeded counts
        self._source_max_exceeded = 0
        self._target_max_exceeded = 0
        with open(cached_path(file_path), "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            
            data_file = list(data_file)

            if bagging:
                print("Executed bagging!!!!!!!!")
                data_file = random.choices(data_file, k = len(data_file))
                
            for line_num, row in enumerate(
                csv.reader(data_file, delimiter=self._delimiter, quoting=self.quoting)
            ):
                if len(row) != 2:
                    raise ConfigurationError(
                        "Invalid line format: %s (line number %d)" % (row, line_num + 1)
                    )
                source_sequence, target_sequence = row
                if len(source_sequence) == 0 or len(target_sequence) == 0:
                    continue

                if self.pseudo:
                    for i in range(len(pseudo_tags)):
                        # pseudo_source_sequence = pseudo_tags[i] + " " + source_sequence
                        # pseudo_target_sequence = pseudo_tags[i] + " " + target_sequence
                        pseudo_source_sequence = source_sequence
                        pseudo_target_sequence = target_sequence
                        ret.append(self.text_to_instance(pseudo_source_sequence, pseudo_target_sequence, v_i=i))
                else:
                    ret.append(self.text_to_instance(source_sequence, target_sequence))
            
            print(f"num of longer than maximux length {self._70}")


        if self._source_max_tokens and self._source_max_exceeded:
            logger.info(
                "In %d instances, the source token length exceeded the max limit (%d) and were truncated.",
                self._source_max_exceeded,
                self._source_max_tokens,
            )
        if self._target_max_tokens and self._target_max_exceeded:
            logger.info(
                "In %d instances, the target token length exceeded the max limit (%d) and were truncated.",
                self._target_max_exceeded,
                self._target_max_tokens,
            )
        
        
        return AllennlpDataset(ret)

    @overrides
    def text_to_instance(
        self, source_string: str, target_string: str = None, v_i = None,
    ) -> Instance:  # type: ignore
        tokenized_source = self._source_tokenizer.tokenize(source_string)
   
        if self._source_max_tokens and len(tokenized_source) > self._source_max_tokens:
            self._source_max_exceeded += 1
            tokenized_source = tokenized_source[: self._source_max_tokens]
        
        if self.pseudo:
            # tokenized_source = [Token(self.tags[v_i])] + tokenized_source
            tokenized_source.insert(0, Token(copy.deepcopy(self.tags[v_i])))


        if self._source_add_start_token:
            tokenized_source.insert(0, Token(copy.deepcopy(self._start_token)))
        if self._source_add_end_token:
            tokenized_source.append(Token(copy.deepcopy(self._end_token)))
        
        self._70 += len(tokenized_source) >= 70

        l_s = len(tokenized_source)//20*20
        self.s_dic[l_s] = self.s_dic.get(l_s, 0) + 1

        source_field = TextField(tokenized_source, self._source_token_indexers)
        if target_string is not None:
            
            tokenized_target = self._target_tokenizer.tokenize(target_string)
 
            if self._target_max_tokens and len(tokenized_target) > self._target_max_tokens:
                self._target_max_exceeded += 1
                tokenized_target = tokenized_target[: self._target_max_tokens]
            if self._target_add_start_token:
                tokenized_target.insert(0, Token(copy.deepcopy(self._start_token)))
            if self._target_add_end_token:
                tokenized_target.append(Token(copy.deepcopy(self._end_token)))
            target_field = TextField(tokenized_target, self._target_token_indexers)
            l_t = len(tokenized_target)//20*20
            self.t_dic[l_t] = self.t_dic.get(l_t, 0) + 1

            return Instance({"source_tokens": source_field, "target_tokens": target_field})
        else:
            return Instance({"source_tokens": source_field})
