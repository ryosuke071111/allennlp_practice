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

# from ensemble import Ensemble

import sys

# cur_dir = os.getcwd()
cur_dir = "/home/acb11356ts/desktop/allennlp_practice"
sys.path.append(os.path.join(os.path.dirname(__file__), f'{cur_dir}/custom_allennlp_components'))
sys.path.append(os.path.join(os.path.dirname(__file__),  f"{cur_dir}/custom_allennlp_components/custom_dataset_reader"))

from custom_allennlp_components.custom_dataset_reader.conll2003_inflated import Conll2003DatasetReader
from custom_allennlp_components.custom_dataset_reader.seq2seq_inflated import PseudoSeq2SeqDatasetReader
from custom_allennlp_components.custom_models.pseudo_crf_tagger import PseudoCrfTagger
from custom_allennlp_components.custom_models.pseudo_composed_seq2seq import PseudoComposedSeq2Seq
from custom_allennlp_components.pseudo_auto_regressive import PseudoAutoRegressiveSeqDecoder
from custom_allennlp_components.sentencepiece_tokenizer import SentencePieceTokenizer
from custom_allennlp_components.inverse_with_warmup import InverseSquareRootLR

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
    source_tokenizer = SentencePieceTokenizer(f"{cur_dir}/iwslt14/spm_en.model")
    target_tokenizer = SentencePieceTokenizer(f"{cur_dir}/iwslt14/spm_de.model")

    # source_tokenizer = WhitespaceTokenizer()
    # target_tokenizer = WhitespaceTokenizer()

    # indexers = {"source_tokens":SingleIdTokenIndexer(), "target_tokens":SingleIdTokenIndexer()}
    source_token_indexers = {"source_tokens":SingleIdTokenIndexer(namespace="source_tokens", lowercase_tokens=True)}
    target_token_indexers = {"target_tokens":SingleIdTokenIndexer(namespace="target_tokens", lowercase_tokens=True)}

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

    ret = Vocabulary()

    ret.set_from_file(filename=f"{cur_dir}/iwslt14/vocab.en",  namespace="source_tokens")
    ret.set_from_file(filename=f"{cur_dir}/iwslt14/vocab.de",  namespace="target_tokens")

    return ret

def build_model(vocab: Vocabulary) -> Model: 
    print("Building the model")
    vocab_size_s = vocab.get_vocab_size("source_tokens")
    vocab_size_t = vocab.get_vocab_size("target_tokens") 
    
    bleu = BLEU(exclude_indices = {0,2,3})

    source_text_embedder = BasicTextFieldEmbedder({"source_tokens": Embedding(embedding_dim=embedding_dim, num_embeddings=vocab_size_s)})
    encoder = PytorchTransformer(input_dim=embedding_dim, num_layers=num_layers ,positional_encoding="sinusoidal", 
                            feedforward_hidden_dim=dff, num_attention_heads=num_head, positional_embedding_size = embedding_dim, dropout_prob = dropout)

    
    # target_text_embedder = BasicTextFieldEmbedder({"target_tokens":Embedding(embedding_dim=embedding_dim, num_embeddings=vocab_size_t)})
    target_text_embedder = Embedding(embedding_dim=embedding_dim, num_embeddings=vocab_size_t)
    decoder_net = StackedSelfAttentionDecoderNet(decoding_dim=embedding_dim, target_embedding_dim=embedding_dim, 
                                feedforward_hidden_dim=dff, num_layers=num_layers, num_attention_heads=num_head, dropout_prob = dropout)
    decoder_net.decodes_parallel=True
    decoder = AutoRegressiveSeqDecoder(
        vocab, decoder_net, max_len, target_text_embedder, 
        target_namespace="target_tokens", tensor_based_metric=bleu, scheduled_sampling_ratio=0.0)
    
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
    # lr_scheduler = InverseSquareRootLR(optimizer, warmup_steps = warmup, end_lr = lr)
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

    print(vocab.get_token_to_index_vocabulary(namespace = "source_tokens"))
    print(vocab.get_token_to_index_vocabulary(namespace = "target_tokens"))

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

# cur_dir = os.getcwd()

TRAIN_PATH = f"{cur_dir}/iwslt14/train"
DEV_PATH = f"{cur_dir}/iwslt14/valid_small"
TEST_PATH = f"{cur_dir}/iwslt14/test"

# TRAIN_PATH = "./data_small/small_japanese"
# DEV_PATH = "./data_small/small_japanese"
# TEST_PATH = "./data_small/small_japanese"


batch_size = 100
embedding_dim = 256
num_layers = 6
dff = 1024
num_head = 4
num_epoch = 600
lr = 5e-4
# num_labels = 2
dropout = 0.3
grad_accum = 1
weight_decay = 0.0001
num_serialized_models_to_keep = 3
grad_norm = None
patience = None
max_len = 70
warmup = 4000

import datetime
now = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())

special = f"_noam_batch{batch_size*grad_accum}_emb{embedding_dim}"
pseudo = "single" if not args.pseudo else "pseudo"
serialization_dir = f"{cur_dir}/checkpoints/nmt_lr_{str(lr)}_{now}_seed{str(seed)}_{pseudo}{special}"

model, reader, trainer = run_training_loop()

