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
from allennlp_models.generation.dataset_readers import Seq2SeqDatasetReader

from allennlp_models.generation.models import ComposedSeq2Seq
from allennlp_models.generation.modules.decoder_nets import StackedSelfAttentionDecoderNet
from allennlp_models.generation.modules.seq_decoders import AutoRegressiveSeqDecoder
from allennlp.training.metrics import BLEU, Entropy
from allennlp.training.learning_rate_schedulers import LinearWithWarmup, NoamLR
from allennlp.training.learning_rate_schedulers.learning_rate_scheduler import ReduceOnPlateauLearningRateScheduler

import torch.nn.functional as F

from ensemble import Ensemble

import sys

cur_dir = os.getcwd()
sys.path.append(os.path.join(os.path.dirname(__file__), cur_dir + '/custom_allennlp_components'))
sys.path.append(os.path.join(os.path.dirname(__file__), cur_dir +  "/custom_allennlp_components/custom_dataset_reader"))

from custom_allennlp_components.custom_dataset_reader.conll2003_inflated import Conll2003DatasetReader
from custom_allennlp_components.custom_dataset_reader.seq2seq_inflated import PseudoSeq2SeqDatasetReader
from custom_allennlp_components.custom_models.pseudo_crf_tagger import PseudoCrfTagger
from custom_allennlp_components.custom_models.pseudo_composed_seq2seq import PseudoComposedSeq2Seq
from custom_allennlp_components.pseudo_auto_regressive import PseudoAutoRegressiveSeqDecoder

import argparse
import random
#
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--pseudo', action="store_true", 
                            help='type pseudo, if you want to activate pseudo ensemble')
parser.add_argument('--dec', action="store_true", 
                            help='type pseudo, if you want to have decoder linear shift embedding')
parser.add_argument('--seed', type=int, default = 42, 
                            help='type seed number')

    
args = parser.parse_args()

seed = args.seed
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

num_virtual_models = 9

tags = ["[pseudo1]","[pseudo2]","[pseudo3]","[pseudo4]","[pseudo5]", "[pseudo6]","[pseudo7]","[pseudo8]","[pseudo9]"][:num_virtual_models]

def build_dataset_reader()  -> DatasetReader:
    source_tokenizer = WhitespaceTokenizer()
    target_tokenizer = WhitespaceTokenizer()

    # indexers = {"source_tokens":SingleIdTokenIndexer(), "target_tokens":SingleIdTokenIndexer()}
    source_token_indexers = {"source_tokens":SingleIdTokenIndexer(namespace="source_tokens", lowercase_tokens=False)}
    target_token_indexers = {"target_tokens":SingleIdTokenIndexer(namespace="target_tokens", lowercase_tokens=False)}

    return PseudoSeq2SeqDatasetReader(source_tokenizer=source_tokenizer, \
                                target_tokenizer=target_tokenizer, \
                                source_token_indexers=source_token_indexers, \
                                target_token_indexers=target_token_indexers, \
                                target_max_tokens=max_len,
                                # start_symbol = "<s>",
                                # end_symbol = "</s>",
                                pseudo = args.pseudo)

def read_data(reader: DatasetReader) -> Tuple[Iterable[Instance], Iterable[Instance]]:
    print("Reading data")
    bagging = False if not args.pseudo else True
    training_data = reader._read(TRAIN_PATH, bagging = bagging)
    validation_data = reader._read(DEV_PATH)
    return training_data, validation_data

def build_vocab(instances: Iterable[Instance]) -> Vocabulary:
    print("Building the vocabulary")
    # ret = Vocabulary.from_instances(instances, min_count={'source_tokens': 3, 'target_tokens': 3})
    # print(ret.get_index_to_token_vocabulary(namespace = "source_tokens"))
    # print(ret.get_index_to_token_vocabulary(namespace = "target_tokens"))
    # exit()
    ret = Vocabulary()
    # ret = ret.from_instances(instances)
    ret.set_from_file(filename="./iwslt15/vocab.en",  namespace="source_tokens")
    ret.set_from_file(filename="./iwslt15/vocab.vi",  namespace="target_tokens")

    print("source vocab length", len(ret.get_index_to_token_vocabulary(namespace = "source_tokens")))
    print("target vocab length", len(ret.get_index_to_token_vocabulary(namespace = "target_tokens")))
    # exit()

    return ret

def build_model(vocab: Vocabulary) -> Model:
    print("Building the model")
    vocab_size_s = vocab.get_vocab_size("source_tokens")
    vocab_size_t = vocab.get_vocab_size("target_tokens")
    
    bleu = BLEU(exclude_indices = {0,2,3})

    source_text_embedder = BasicTextFieldEmbedder({"source_tokens": Embedding(embedding_dim=embedding_dim, num_embeddings=vocab_size_s)})
    encoder = PytorchTransformer(input_dim=embedding_dim, num_layers=num_layers ,positional_encoding="sinusoidal", feedforward_hidden_dim=dff, num_attention_heads=num_head)

    
    # target_text_embedder = BasicTextFieldEmbedder({"target_tokens":Embedding(embedding_dim=embedding_dim, num_embeddings=vocab_size_t)})
    target_text_embedder = Embedding(embedding_dim=embedding_dim, num_embeddings=vocab_size_t)
    decoder_net = StackedSelfAttentionDecoderNet(decoding_dim=embedding_dim, target_embedding_dim=embedding_dim, feedforward_hidden_dim=dff, num_layers=num_layers, num_attention_heads=num_head)
    decoder_net.decodes_parallel=True
    decoder = AutoRegressiveSeqDecoder(
        vocab, decoder_net, max_len, target_text_embedder, 
        target_namespace="target_tokens", tensor_based_metric=bleu, 
        scheduled_sampling_ratio=0.0)
    
    if args.pseudo:
        decoder = PseudoAutoRegressiveSeqDecoder(vocab, decoder_net, max_len, target_text_embedder, target_namespace="target_tokens", tensor_based_metric=bleu, scheduled_sampling_ratio=0.0, decoder_lin_emb = args.dec)
        return PseudoComposedSeq2Seq(vocab, source_text_embedder, encoder, decoder, num_virtual_models = num_virtual_models)
    else:
        decoder = AutoRegressiveSeqDecoder(vocab, decoder_net, max_len, target_text_embedder, target_namespace="target_tokens", tensor_based_metric=bleu, scheduled_sampling_ratio=0.0)
        return ComposedSeq2Seq(vocab, source_text_embedder, encoder, decoder)

def build_data_loaders(train_data: torch.utils.data.Dataset, dev_data: torch.utils.data.Dataset) -> Tuple[allennlp.data.PyTorchDataLoader, allennlp.data.PyTorchDataLoader]:
    train_loader = PyTorchDataLoader(train_data, batch_size=batch_size, shuffle=True)
    dev_loader = PyTorchDataLoader(dev_data, batch_size=num_virtual_models, shuffle=False)
    return train_loader, dev_loader

def build_trainer(model: Model, serialization_dir:str, train_loader: PyTorchDataLoader, dev_loader: PyTorchDataLoader) -> Trainer:
    parameters = [[n,p] for n, p in model.named_parameters() if p.requires_grad]
    optimizer = AdamOptimizer(parameters, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.98), eps=1e-09)
    lr_scheduler = NoamLR(optimizer, model_size = embedding_dim, warmup_steps = warmup)
    # lr_scheduler = ReduceOnPlateauLearningRateScheduler(optimizer, factor = 0.8, patience = 3, min_lr = 0.000001, eps=1e-08)
    trainer = GradientDescentTrainer(
        model=model, 
        serialization_dir=serialization_dir, 
        data_loader=train_loader, \
        validation_data_loader=dev_loader, 
        num_epochs=num_epoch, 
        optimizer=optimizer, \
        num_gradient_accumulation_steps=grad_accum,
        grad_norm=grad_norm, 
        patience=patience,
        learning_rate_scheduler=lr_scheduler)
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

    # with tempfile.TemporaryDirectory() as serialization_dir:
    trainer = build_trainer(model, serialization_dir, train_loader, dev_loader)
    print("Starting training")
    trainer.train()
    print("Finished training")

    return model, dataset_reader, trainer

cur_dir = os.getcwd()
TRAIN_PATH = cur_dir + "/wmt/train"
DEV_PATH = cur_dir + "/wmt/test"
TEST_PATH = cur_dir + "/wmt/test"

TRAIN_PATH = "./iwslt15/train"
DEV_PATH = "./iwslt15/valid"
TEST_PATH = "./iwslt15/test"

# TRAIN_PATH = "./data_small/small_japanese"
# DEV_PATH = "./data_small/small_japanese"
# TEST_PATH = "./data_small/small_japanese"


batch_size = 32
embedding_dim = 256
num_layers = 2
dff = 1024
num_head = 4
num_epoch = 75
lr = 1e-3
# num_labels = 2
grad_accum = 1
weight_decay = 0.00001
num_serialized_models_to_keep = 3
grad_norm = 5.0
patience = None
max_len = 70
warmup = 4000

import datetime
now = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())

serialization_dir = "./checkpoints/nmt_lr_" + str(lr) + "_" + now + "_seed" + str(seed) + "_" + ("single" if not args.pseudo else "pseudo")

model, reader, trainer = run_training_loop()

# dataset_reader = build_dataset_reader()
# train_data, dev_data = read_data(dataset_reader)
# train_loader, dev_loader = build_data_loaders(train_data, dev_data)
# vocab = build_vocab(train_data + dev_data)
# train_data.index_with(vocab)
# dev_data.index_with(vocab)


# model1.cuda() if torch.cuda.is_available() else model1


# model2.cuda() if torch.cuda.is_available() else model2

# model1 = build_model(vocab)
# model2 = build_model(vocab)
# models = [model1, model2]
# ensemble = Ensemble(models)
# for data in train_loader:
#     # data.to("cuda")
#     print(ensemble(data["source_tokens"], data["target_tokens"]))






# test_data = dataset_reader.read(TEST_PATH)
# test_data.index_with(model.vocab)
# data_loader = PyTorchDataLoader(test_data, batch_size=32)


# results = evaluate(model, data_loader, cuda_device=0)
# print(results)
# print("batch_size:{}, num_epoch:{}, lr:{}, grad_accum:{}".format(batch_size, num_epoch, lr, grad_accum))


#----------------------------------------------------------------------
