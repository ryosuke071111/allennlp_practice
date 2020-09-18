import tempfile
from typing import Dict, Iterable, List, Tuple
import os

import allennlp
import torch
from allennlp.data import PyTorchDataLoader, DatasetReader, Instance, Vocabulary
from allennlp.data.dataset_readers import SequenceTaggingDatasetReader, Conll2003DatasetReader
from allennlp.data.fields import LabelField, TextField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer, PretrainedTransformerIndexer, ELMoTokenCharactersIndexer, TokenCharactersIndexer, PretrainedTransformerMismatchedIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WhitespaceTokenizer, PretrainedTransformerTokenizer
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder
from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder, CnnEncoder
from allennlp.modules.seq2seq_encoders import PytorchTransformer
from allennlp.modules.token_embedders import Embedding, TokenEmbedder, PretrainedTransformerEmbedder, ElmoTokenEmbedder, TokenCharactersEncoder, PretrainedTransformerMismatchedEmbedder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.nn import util
from allennlp.training.trainer import GradientDescentTrainer, Trainer
from allennlp.training.optimizers import AdamOptimizer
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.training.util import evaluate
from allennlp.predictors import Predictor
from allennlp.common import JsonDict
from allennlp.modules.seq2vec_encoders.cnn_encoder import CnnEncoder
from allennlp.modules.seq2vec_encoders.bert_pooler import BertPooler
from allennlp.modules.seq2seq_encoders import LstmSeq2SeqEncoder

from transformers import BertTokenizer

from allennlp_models.tagging.models import CrfTagger

         

def build_dataset_reader()  -> DatasetReader:
    token_indexers =  {"tokens":SingleIdTokenIndexer(lowercase_tokens=True), "elmo":ELMoTokenCharactersIndexer(), "token_characters":TokenCharactersIndexer(min_padding_length=3)}
    # token_indexers = {"tokens":PretrainedTransformerMismatchedIndexer("bert-large-uncased", max_length=512)}

    tag_label = "ner"
    # coding_scheme = "BIOUL" #BIOUL for elmo IOB1 for elmo
    coding_scheme = "BIOUL"
    return Conll2003DatasetReader(token_indexers=token_indexers, tag_label=tag_label, coding_scheme=coding_scheme)

def read_data(reader: DatasetReader) -> Tuple[Iterable[Instance], Iterable[Instance]]:
    print("Reading data")
    training_data = reader.read(TRAIN_PATH)
    validation_data = reader.read(DEV_PATH)
    return training_data, validation_data

def build_vocab(instances: Iterable[Instance]) -> Vocabulary:
    print("Building the vocabulary")
    return Vocabulary.from_instances(instances)

def build_model(vocab: Vocabulary) -> Model:
    print("Building the model")
    vocab_size_tokens = vocab.get_vocab_size("tokens")
    vocab_size_chars = vocab.get_vocab_size("token_characters")

    embedder = BasicTextFieldEmbedder({"tokens": Embedding(embedding_dim=embedding_dim, pretrained_file="/home/ryosuke/desktop/allen_practice/glove/glove.6B.200d.txt", trainable=True, num_embeddings=vocab_size_tokens, vocab=vocab),\
                                        "elmo": ElmoTokenEmbedder(weight_file="https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5", do_layer_norm=False, dropout=0.0),\
                                        "token_characters":TokenCharactersEncoder(embedding=Embedding(embedding_dim=16, num_embeddings=vocab_size_chars, vocab=vocab), encoder=CnnEncoder(embedding_dim=16, num_filters=128, ngram_filter_sizes=[3]))})
    encoder = PytorchTransformer(input_dim=1352, num_layers=6, positional_encoding="sinusoidal")

    # embedder = BasicTextFieldEmbedder({"tokens": Embedding(embedding_dim=embedding_dim, num_embeddings=vocab_size)})
    # encoder = BagOfEmbeddingsEncoder(embedding_dim=embedding_dim)
    # embedder = BasicTextFieldEmbedder({"tokens": PretrainedTransformerMismatchedEmbedder("bert-large-uncased")})
    # encoder = LstmSeq2SeqEncoder(input_size=1024, hidden_size=1024, num_layers=2, dropout=0.5, bidirectional=True)


    
    return CrfTagger(vocab, embedder, encoder, \
                label_encoding="BIOUL", include_start_end_transitions=False)

def build_data_loaders(train_data: torch.utils.data.Dataset, dev_data: torch.utils.data.Dataset) -> Tuple[allennlp.data.PyTorchDataLoader, allennlp.data.PyTorchDataLoader]:
    train_loader = PyTorchDataLoader(train_data, batch_size=batch_size, shuffle=True)
    dev_loader = PyTorchDataLoader(dev_data, batch_size=batch_size, shuffle=True)
    return train_loader, dev_loader

def build_trainer(model: Model, serialization_dir:str, train_loader: PyTorchDataLoader, dev_loader: PyTorchDataLoader) -> Trainer:
    parameters = [[n,p] for n, p in model.named_parameters() if p.requires_grad]
    optimizer = AdamOptimizer(parameters, lr=lr, weight_decay=weight_decay)
    trainer = GradientDescentTrainer(model=model, serialization_dir=serialization_dir, data_loader=train_loader, \
                                validation_data_loader=dev_loader, num_epochs=num_epoch, optimizer=optimizer, \
                                num_gradient_accumulation_steps=grad_accum,
                                validation_metric=validation_metric, grad_norm=grad_norm, patience=patience)
    return trainer





def run_training_loop():
    dataset_reader = build_dataset_reader()
    train_data, dev_data = read_data(dataset_reader)

    vocab = build_vocab(train_data + dev_data)


    model = build_model(vocab)
    model.cuda() if torch.cuda.is_available() else model

    train_data.index_with(vocab)
    dev_data.index_with(vocab)

    train_loader, dev_loader = build_data_loaders(train_data, dev_data)

    with tempfile.TemporaryDirectory() as serialization_dir:
        trainer = build_trainer(model, serialization_dir, train_loader, dev_loader)
        print("Starting training")
        trainer.train()
        print("Finished training")
    
    return model, dataset_reader


TRAIN_PATH = "/home/ryosuke/desktop/allen_practice/conll2003/eng.train"
DEV_PATH = "/home/ryosuke/desktop/allen_practice/conll2003/eng.testb"
TEST_PATH = "/home/ryosuke/desktop/allen_practice/conll2003/eng.testb"

cur_dir = os.getcwd()
TRAIN_PATH = cur_dir + "/conll2003/eng.train"
DEV_PATH = cur_dir + "/conll2003/eng.testb"
TEST_PATH = cur_dir + "/conll2003/eng.testb"

"""
BERT fine-tune hyper parameter
How to Fine-Tune BERT for Text Classification?: https://arxiv.org/pdf/1905.05583.pdf
lr:2e-5
batch:32
"""

batch_size = 2
embedding_dim = 200
num_epoch = 75
lr = 0.00002
num_labels = 2
grad_accum = 16
weight_decay = 0.0001
validation_metric = "+f1-measure-overall"
num_serialized_models_to_keep = 3
grad_norm = 5.0
patience = 25


model, dataset_reader = run_training_loop()
test_data = dataset_reader.read(TEST_PATH)
test_data.index_with(model.vocab)
data_loader = PyTorchDataLoader(test_data, batch_size=2)


results = evaluate(model, data_loader, cuda_device=0)
print(results)
print("batch_size:{}, num_epoch:{}, lr:{}, grad_accum:{}".format(batch_size, num_epoch, lr, grad_accum))
