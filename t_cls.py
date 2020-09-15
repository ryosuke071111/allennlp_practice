import tempfile
from typing import Dict, Iterable, List, Tuple
import os

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

from transformers import BertTokenizer

class ClassificaionTsvReader(DatasetReader):
    def __init__(self, lazy: bool = False, tokenizer: Tokenizer = None, token_indexers: Dict[str, TokenIndexer] = None, max_tokens: int = None):
        super().__init__(lazy)
        # self.tokenizer = tokenizer or WhitespaceTokenizer()
        # self.token_indexers = token_indexers or {"tokens":SingleIdTokenIndexer()}
        self.tokenizer = tokenizer or PretrainedTransformerTokenizer("bert-large-uncased")
        self.token_indexers = token_indexers or {"tokens":PretrainedTransformerIndexer("bert-large-uncased")}
        self.max_tokens = max_tokens
    
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
            for line in list(lines)[:100]:
                text, sentiment = line.strip().split("\t")
                yield self.text_to_instance(text, sentiment)


class SimpleClassifier(Model):
    def __init__(self, vocab: Vocabulary, embedder: TextFieldEmbedder, encoder: Seq2VecEncoder):
        super().__init__(vocab)
        self.embedder = embedder
        self.encoder = encoder
        # num_labels = vocab.get_vocab_size("labels") or 2
        self.classifier = torch.nn.Linear(encoder.get_output_dim(), num_labels)
        self.accuracy = CategoricalAccuracy()

    def forward(self, text: Dict[str, torch.Tensor], label: torch.Tensor = None) -> Dict[str, torch.Tensor]:

        mask = util.get_text_field_mask(text)
        embedded_text = self.embedder(text)
        encoded_text = self.encoder(embedded_text, mask)

        logits = self.classifier(encoded_text)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        loss = torch.nn.functional.cross_entropy(logits, label)
        output = {"probs":probs}

        if label is not None:
            self.accuracy(logits, label)
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
    embedder = BasicTextFieldEmbedder({"tokens": PretrainedTransformerEmbedder("bert-large-uncased")})
    encoder = BertPooler("bert-large-uncased")
    
    # encoder = CnnEncoder(embedding_dim=embedding_dim, num_filters=3, output_dim=embedding_dim)
    
    return SimpleClassifier(vocab, embedder, encoder)

def build_data_loaders(train_data: torch.utils.data.Dataset, dev_data: torch.utils.data.Dataset) -> Tuple[allennlp.data.PyTorchDataLoader, allennlp.data.PyTorchDataLoader]:
    train_loader = PyTorchDataLoader(train_data, batch_size=batch_size, shuffle=True)
    dev_loader = PyTorchDataLoader(dev_data, batch_size=batch_size, shuffle=True)
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

batch_size = 2
embedding_dim = 256
num_epoch = 10
lr = 0.00002
num_labels = 2
grad_accum = 16

model, dataset_reader = run_training_loop()
test_data = dataset_reader.read(TEST_PATH)
test_data.index_with(model.vocab)
data_loader = PyTorchDataLoader(test_data, batch_size=2)


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
