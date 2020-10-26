import tempfile
from typing import Dict, Iterable, List, Tuple
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '/home/ryosuke/desktop/allen_practice/custom_allennlp_components'))

import allennlp
import torch
from allennlp.data import PyTorchDataLoader, DatasetReader, Instance, Vocabulary
from allennlp.data.fields import LabelField, TextField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer, PretrainedTransformerIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WhitespaceTokenizer, PretrainedTransformerTokenizer
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder
from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder
from allennlp.modules.token_embedders import Embedding, TokenEmbedder, PretrainedTransformerEmbedder
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
from allennlp.modules.seq2seq_encoders import PytorchTransformer

from transformers import BertTokenizer, BertModel

from custom_allennlp_components.custom_models.embedding_bert import EmbeddingBertTransformer

import argparse

import random




parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--pseudo', action="store_true", 
                            help='type pseudo, if you want to activate pseudo ensemble')
parser.add_argument('--seed', type=int, default = 42, 
                            help='type seed number')
args = parser.parse_args()

seed = args.seed
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

print("set random seed to {}".format(seed))

def freeze(model):
    for param in model.parameters():
        param.requires_grad = False
    return model

"""
single-ensembleする時は、
- torch.randn(9,256) -> torch.nn.init.orthogonal(d) で線形移動ベクトル作れる
"""
 
num_virtual_models = (5 if args.pseudo else 0)

print("num_virtual_models {}".format(num_virtual_models))

tags = ["[pseudo1]","[pseudo2]","[pseudo3]","[pseudo4]","[pseudo5]", "[pseudo6]","[pseudo7]","[pseudo8]","[pseudo9]"][:num_virtual_models]
# offset = 1646

#平仮名を入れてしまうと、pret-rainedの時の文脈が考慮されているはず

class ClassificaionTsvReader(DatasetReader):
    def __init__(self, lazy: bool = False, tokenizer: Tokenizer = None, token_indexers: Dict[str, TokenIndexer] = None, max_tokens: int = None, pseudo: bool = False if args.pseudo == False else True):
        super().__init__(lazy)
        # self.tokenizer = tokenizer or WhitespaceTokenizer()
        # self.token_indexers = token_indexers or {"tokens":SingleIdTokenIndexer()}
        # from collections import defaultdict
        # ags = defaultdict(list) 
        ags = {"additional_special_tokens":("[pseudo1]","[pseudo2]","[pseudo3]","[pseudo4]","[pseudo5]" , "[pseudo6]","[pseudo7]","[pseudo8]","[pseudo9]")}
        self.tokenizer = tokenizer or PretrainedTransformerTokenizer("bert-large-uncased", tokenizer_kwargs=(ags if args.pseudo else {}))
        self.token_indexers = token_indexers or {"tokens":PretrainedTransformerIndexer("bert-large-uncased")}
        self.max_tokens = max_tokens
        self.pseudo = pseudo
        self.tags = tags

    
    def text_to_instance(self, text:str, label:str = None) -> Instance:
        tokens = self.tokenizer.tokenize(text)

        if self.max_tokens:
            tokens = tokens[:self.max_tokens]
        text_field = TextField(tokens, self.token_indexers)

        fields = {"text":text_field}
        if label:
            fields["label"] = LabelField(label)
            return Instance(fields)

    def _read(self, file_path:str) -> Iterable[Instance]:
        with open(file_path, "r") as lines:
            for line in list(lines):
                text, sentiment = line.strip().split("\t")
                if self.pseudo:
                    for tag in self.tags:
                        tagged_text = tag + " " + text
                        yield self.text_to_instance(tagged_text, sentiment)
                else:
                    yield self.text_to_instance(text, sentiment)


class SimpleClassifier(Model):
    def __init__(self, vocab: Vocabulary, embedder: TextFieldEmbedder, encoder: Seq2VecEncoder or Seq2SeqEncoder, bert_as_embed: bool = True, pseudo: bool = False if args.pseudo == False else True):
        super().__init__(vocab)
        self.pseudo = pseudo

        self.embedder = embedder
        self.encoder = encoder
        self.classifier = torch.nn.Linear(encoder.get_output_dim(), num_labels)
        self.accuracy = CategoricalAccuracy()
        self.bert_as_embed = bert_as_embed
        if bert_as_embed:
            self.bert = freeze(BertModel.from_pretrained("bert-large-uncased"))
            print("Use bert as embedding")
        else:
            self.bert_as_embed = None
        self.device = next(self.bert.parameters()).device

        

        if self.pseudo:
            self.bert.resize_token_embeddings(30522 + num_virtual_models)
            print("resized bert num_embedding to {}".format(self.bert.vocab_size))
            self.vectors = torch.nn.init.orthogonal_(torch.randn(num_virtual_models, 1024, requires_grad = False))
            ags = {"additional_special_tokens":("[pseudo1]","[pseudo2]","[pseudo3]","[pseudo4]","[pseudo5]" , "[pseudo6]","[pseudo7]","[pseudo8]","[pseudo9]")}
            self.tokenizer = PretrainedTransformerTokenizer("bert-large-uncased", tokenizer_kwargs=ags)

        self.vocab = vocab
        
        
        

    def forward(self, text: Dict[str, torch.Tensor], label: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        device = next(self.bert.parameters()).device

        # if self.pseudo:
        text["tokens"]["tokens"] = text["tokens"]["token_ids"]
        del text["tokens"]["token_ids"] 
        del text["tokens"]["type_ids"]
        del text["tokens"]["mask"]

        embedded_text = self.embedder(text)
        if self.pseudo:
            offset = 30522
            vector_ids = text["tokens"]["tokens"][:, 1] - offset
            print(text["tokens"]["tokens"][:,1])
            # print(text["tokens"]["tokens"][:,1] -offset)
            # exit()
            vector_ids = text["tokens"]["tokens"][:, 1] - offset
            vectors = torch.index_select(self.vectors.to(device) , 0, vector_ids).to(device).unsqueeze(1)
            embedded_text += vectors

        mask = util.get_text_field_mask(text)

        if self.bert_as_embed:
            bert = self.bert(text["tokens"]["tokens"])[0]
            encoded_text = self.encoder(embedded_text, mask, bert)
        else:
            encoded_text = self.encoder(embedded_text, mask)

        logits = self.classifier(encoded_text)
        probs = torch.nn.functional.softmax(logits, dim=-1)

        if not self.training and self.pseudo:
            probs = torch.mean(probs, 0).unsqueeze(0)
            logits = torch.mean(logits, 0).unsqueeze(0)
            label = label[0].unsqueeze(0)


        loss = torch.nn.functional.cross_entropy(logits, label)
        output = {"probs":probs}

        if label is not None:
            self.accuracy(probs, label)
            output["loss"] = torch.nn.functional.cross_entropy(logits, label)

        return output
    
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}

         

def build_dataset_reader()  -> DatasetReader:
    return ClassificaionTsvReader(max_tokens=512)

def read_data(reader: DatasetReader) -> Tuple[Iterable[Instance], Iterable[Instance]]:
    print("Reading data")
    training_data = reader.read(TRAIN_PATH)
    validation_data = reader.read(DEV_PATH)
    return training_data, validation_data

def build_vocab(instances: Iterable[Instance]) -> Vocabulary:
    print("Building the vocabulary")
    # import pdb;pdb.set_trace()
    return Vocabulary.from_instances(instances)

def build_model(vocab: Vocabulary) -> Model:
    print("Building the model")
    vocab_size = vocab.get_vocab_size("tokens")

    # embedder = BasicTextFieldEmbedder({"tokens": Embedding(embedding_dim=embedding_dim, num_embeddings=vocab_size)})
    # encoder = BagOfEmbeddingsEncoder(embedding_dim=embedding_dim)
    # encoder = CnnEncoder(embedding_dim=embedding_dim, num_filters=3, output_dim=embedding_dim)

    #BERT fine tune
    # embedder = BasicTextFieldEmbedder({"tokens": PretrainedTransformerEmbedder("bert-large-uncased")})
    # encoder = BertPooler("bert-large-uncased")
    # return SimpleClassifier(vocab, embedder, encoder)

    #Embeddin BERT
    embedder = BasicTextFieldEmbedder({"tokens": Embedding(embedding_dim=1024, num_embeddings=30522 + num_virtual_models)})
    encoder = EmbeddingBertTransformer(input_dim=1024, num_layers=6, positional_encoding="sinusoidal")
    return SimpleClassifier(vocab, embedder, encoder, bert_as_embed=True)


def build_data_loaders(train_data: torch.utils.data.Dataset, dev_data: torch.utils.data.Dataset) -> Tuple[allennlp.data.PyTorchDataLoader, allennlp.data.PyTorchDataLoader]:

    train_loader = PyTorchDataLoader(train_data, batch_size=batch_size, shuffle=True)
    dev_loader = PyTorchDataLoader(dev_data, batch_size=batch_size, shuffle=False)
    return train_loader, dev_loader

def build_trainer(model: Model, serialization_dir:str, train_loader: PyTorchDataLoader, dev_loader: PyTorchDataLoader) -> Trainer:
    parameters = [[n,p] for n, p in model.named_parameters() if p.requires_grad]
    optimizer = AdamOptimizer(parameters, lr=lr)
    trainer = GradientDescentTrainer(model=model, serialization_dir=serialization_dir, data_loader=train_loader, \
                validation_data_loader=dev_loader, num_epochs=num_epoch, optimizer=optimizer, num_gradient_accumulation_steps=grad_accum)
    return trainer

class SentenceClassifierPredictor(Predictor):
    def predict(self, sentence: str) -> JsonDict:
        return self.predict_json({"sentence": sentence})

    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        sentence = json_dict["sentence"]
        return self._dataset_reader.text_to_instance(sentence)


def run_training_loop():
    dataset_reader = build_dataset_reader()
    train_data, dev_data = read_data(dataset_reader)

    vocab = build_vocab(train_data + dev_data)

    if not os.path.exists(vocab_dir):
        vocab.save_to_files("./")
    else:
        raise Exception("vocabulary serialization directory %s is not empty", vocab_dir)
        exit()


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
    
    return model, dataset_reader

cur_dir = os.getcwd()
TRAIN_PATH = cur_dir + "/imdb/train"
DEV_PATH = cur_dir + "/imdb/test"
TEST_PATH = cur_dir + "/imdb/test"

# TRAIN_PATH = "/home/ryosuke/desktop/allen_practice/train.tsv"
# DEV_PATH = "/home/ryosuke/desktop/allen_practice/train.tsv"
# TEST_PATH = "/home/ryosuke/desktop/allen_practice/train.tsv"

"""
BERT fine-tune hyper parameter
How to Fine-Tune BERT for Text Classification?: https://arxiv.org/pdf/1905.05583.pdf
lr:2e-5
batch:32
"""

batch_size = 4
embedding_dim = 256
num_epoch = 100
lr = 0.00002
num_labels = 2
grad_accum = 8

import datetime
now = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())

serialization_dir = "/home/ryosuke/desktop/allen_practice/checkpoints_clss/lr_" + str(lr) + "_" + now + "_seed" + str(seed) + "_" + ("single" if not args.pseudo else "pseudo")
vocab_dir = serialization_dir + "/vocab"

model, dataset_reader = run_training_loop()
test_data = dataset_reader.read(TEST_PATH)
test_data.index_with(model.vocab)
data_loader = PyTorchDataLoader(test_data, batch_size=batch_size, shuffle=False)


results = evaluate(model, data_loader, cuda_device=0)
print(results)
print("batch_size:{}, num_epoch:{}, lr:{}, grad_accum:{}".format(batch_size, num_epoch, lr, grad_accum))

# vocab = model.vocab
# predictor = SentenceClassifierPredictor(model, dataset_reader)

# output = predictor.predict('A good movie!')
# print([(vocab.get_token_from_index(label_id, 'labels'), prob)
#        for label_id, prob in enumerate(output['probs'])])
# output = predictor.predict('This was a monstrous waste of time.')
# print([(vocab.get_token_from_index(label_id, 'labels'), prob)
#        for label_id, prob in enumerate(output['probs'])])transformers

