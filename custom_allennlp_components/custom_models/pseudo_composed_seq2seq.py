from typing import Dict, Optional

import torch
from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder, Embedding
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.nn import util, InitializerApplicator

from allennlp_models.generation.modules.seq_decoders.seq_decoder import SeqDecoder


@Model.register("pseudocomposed_seq2seq")
class PseudoComposedSeq2Seq(Model):
    """
    This `ComposedSeq2Seq` class is a `Model` which takes a sequence, encodes it, and then
    uses the encoded representations to decode another sequence.  You can use this as the basis for
    a neural machine translation system, an abstractive summarization system, or any other common
    seq2seq problem.  The model here is simple, but should be a decent starting place for
    implementing recent models for these tasks.

    The `ComposedSeq2Seq` class composes separate `Seq2SeqEncoder` and `SeqDecoder` classes.
    These parts are customizable and are independent from each other.

    # Parameters

    vocab : `Vocabulary`, required
        Vocabulary containing source and target vocabularies. They may be under the same namespace
        (`tokens`) or the target tokens can have a different namespace, in which case it needs to
        be specified as `target_namespace`.
    source_text_embedders : `TextFieldEmbedder`, required
        Embedders for source side sequences
    encoder : `Seq2SeqEncoder`, required
        The encoder of the "encoder/decoder" model
    decoder : `SeqDecoder`, required
        The decoder of the "encoder/decoder" model
    tied_source_embedder_key : `str`, optional (default=`None`)
        If specified, this key is used to obtain token_embedder in `source_text_embedder` and
        the weights are shared/tied with the decoder's target embedding weights.
    initializer : `InitializerApplicator`, optional (default=`InitializerApplicator()`)
        Used to initialize the model parameters.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        source_text_embedder: TextFieldEmbedder,
        encoder: Seq2SeqEncoder,
        decoder: SeqDecoder,
        tied_source_embedder_key: Optional[str] = None,
        initializer: InitializerApplicator = InitializerApplicator(),
        num_virtual_models: int = 9,
        **kwargs,
    ) -> None:

        super().__init__(vocab, **kwargs)

        self._source_text_embedder = source_text_embedder
        self._encoder = encoder
        self._decoder = decoder

        if self._encoder.get_output_dim() != self._decoder.get_output_dim():
            raise ConfigurationError(
                f"Encoder output dimension {self._encoder.get_output_dim()} should be"
                f" equal to decoder dimension {self._decoder.get_output_dim()}."
            )
        if tied_source_embedder_key:
            # A bit of a ugly hack to tie embeddings.
            # Works only for `BasicTextFieldEmbedder`, and since
            # it can have multiple embedders, and `SeqDecoder` contains only a single embedder, we need
            # the key to select the source embedder to replace it with the target embedder from the decoder.
            if not isinstance(self._source_text_embedder, BasicTextFieldEmbedder):
                raise ConfigurationError(
                    "Unable to tie embeddings,"
                    "Source text embedder is not an instance of `BasicTextFieldEmbedder`."
                )

            source_embedder = self._source_text_embedder._token_embedders[tied_source_embedder_key]
            if not isinstance(source_embedder, Embedding):
                raise ConfigurationError(
                    "Unable to tie embeddings,"
                    "Selected source embedder is not an instance of `Embedding`."
                )
            if source_embedder.get_output_dim() != self._decoder.target_embedder.get_output_dim():
                raise ConfigurationError(
                    "Output Dimensions mismatch between source embedder and target embedder."
                )
            self._source_text_embedder._token_embedders[
                tied_source_embedder_key
            ] = self._decoder.target_embedder
        
        self.num_virtual_models = num_virtual_models
        self.index_dict = {"[pseudo1]":0, "[pseudo2]":1, "[pseudo3]":2, "[pseudo4]":3, "[pseudo5]":4, "[pseudo6]":5, "[pseudo7]":6, "[pseudo8]":7, "[pseudo9]":8}

        self.orthogonal_embedding_emb = torch.nn.init.orthogonal_(torch.empty(self.num_virtual_models, source_text_embedder.get_output_dim(),requires_grad=False)).float()
        # self.orthogonal_embedding_hidden = torch.nn.init.orthogonal_(torch.empty(self.num_virtual_models, encoder.get_output_dim(),requires_grad=False)).float()
        
        self.vocab = vocab

        initializer(self)

    @overrides
    def forward(
        self,  # type: ignore
        source_tokens: TextFieldTensors,
        target_tokens: TextFieldTensors = None,
    ) -> Dict[str, torch.Tensor]:

        """
        Make forward pass on the encoder and decoder for producing the entire target sequence.

        # Parameters

        source_tokens : `TextFieldTensors`
           The output of `TextField.as_array()` applied on the source `TextField`. This will be
           passed through a `TextFieldEmbedder` and then through an encoder.
        target_tokens : `TextFieldTensors`, optional (default = `None`)
           Output of `Textfield.as_array()` applied on target `TextField`. We assume that the
           target tokens are also represented as a `TextField`.

        # Returns

        `Dict[str, torch.Tensor]`
            The output tensors from the decoder.
        """



        state = self._encode(source_tokens)

        ret = self._decoder(state, target_tokens, source_tokens)

        return ret

    @overrides
    def make_output_human_readable(
        self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Finalize predictions.
        """
        return self._decoder.post_process(output_dict)

    def _encode(self, source_tokens: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Make foward pass on the encoder.

        # Parameters

        source_tokens : `TextFieldTensors`
           The output of `TextField.as_array()` applied on the source `TextField`. This will be
           passed through a `TextFieldEmbedder` and then through an encoder.

        # Returns

        Dict[str, torch.Tensor]
            Map consisting of the key `source_mask` with the mask over the
            `source_tokens` text field,
            and the key `encoder_outputs` with the output tensor from
            forward pass on the encoder.
        """

        # print(self.vocab.get_index_to_token_vocabulary(namespace = "source_tokens"))
        # exit()
        
        print("sour",source_tokens)
        try:
            # print("tokens", tokens["tokens"])
            index = torch.tensor([self.index_dict[self.vocab.get_token_from_index(i, namespace = "source_tokens")] for i in source_tokens["source_tokens"]["tokens"][:,1].squeeze(0).tolist()]).long().to(source_tokens["source_tokens"]["tokens"].device)
            # print(tokens["tokens"][:,0])
            # exit()
            # index = torch.tensor([self.index_dict[self.vocab.get_token_from_index(i)] for i in tokens["tokens"][:,0].squeeze(0).tolist()]).long().to(tokens["tokens"].device)
        except:
            import sys
            sys.stdout.write("AAA {}".format(sys.exc_info()))
            index = torch.tensor([self.index_dict[self.vocab.get_token_from_index(i,  namespace = "source_tokens")] for i in source_tokens["source_tokens"]["tokens"][:,1].squeeze(0).tolist()]).long().to(source_tokens["source_tokens"]["tokens"].device)


        # shape: (batch_size, max_input_sequence_length, encoder_input_dim)
        embedded_input = self._source_text_embedder(source_tokens)

        # # orthogonal embedding
        orthogonal_vecs = torch.index_select(self.orthogonal_embedding_emb.to(embedded_input.device),0,index).unsqueeze(1)
        bsz, seq_len, emb_dim = embedded_input.size()
        orthogonal_vecs = orthogonal_vecs.expand(-1, seq_len, -1)
        
        scale = 1

        # embedded_text_input = F.normalize(embedded_text_input,dim=-1)

        embedded_input += (scale*orthogonal_vecs)

        # shape: (batch_size, max_input_sequence_length)()
        source_mask = util.get_text_field_mask(source_tokens)
        # shape: (batch_size, max_input_sequence_length, encoder_output_dim)
        encoder_outputs = self._encoder(embedded_input, source_mask)
        return {"source_mask": source_mask, "encoder_outputs": encoder_outputs}

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return self._decoder.get_metrics(reset)

    default_predictor = "seq2seq"
