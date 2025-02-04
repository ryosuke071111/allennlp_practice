from typing import Dict, List, Tuple, Optional
from overrides import overrides

import numpy
import torch
import torch.nn.functional as F
from torch.nn import Linear

from allennlp.common.checks import ConfigurationError
from allennlp.common.util import END_SYMBOL, START_SYMBOL
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.modules import Embedding
from allennlp.nn import util
from allennlp.nn.beam_search import BeamSearch
from allennlp.training.metrics import Metric

from allennlp_models.generation.modules.decoder_nets.decoder_net import DecoderNet
from allennlp_models.generation.modules.seq_decoders import SeqDecoder

num_virtual_models = 9

# START_SYMBOL = "<s>"
# END_SYMBOL = "</s>"

def reconstruct_sequences(predictions, backpointers):
    # Reconstruct the sequences.
    # shape: [(batch_size, beam_size, 1)]
    reconstructed_predictions = [predictions[-1].unsqueeze(2)]

    # shape: (batch_size, beam_size)
    cur_backpointers = backpointers[-1]

    for timestep in range(len(predictions) - 2, 0, -1):
        # shape: (batch_size, beam_size, 1)
        cur_preds = predictions[timestep].gather(1, cur_backpointers).unsqueeze(2)

        reconstructed_predictions.append(cur_preds)

        # shape: (batch_size, beam_size)
        cur_backpointers = backpointers[timestep - 1].gather(1, cur_backpointers)

    # shape: (batch_size, beam_size, 1)
    final_preds = predictions[0].gather(1, cur_backpointers).unsqueeze(2)

    reconstructed_predictions.append(final_preds)

    return reconstructed_predictions

@SeqDecoder.register("pseudo_auto_regressive_seq_decoder")
class PseudoAutoRegressiveSeqDecoder(SeqDecoder):
    """
    An autoregressive decoder that can be used for most seq2seq tasks.

    # Parameters

    vocab : `Vocabulary`, required
        Vocabulary containing source and target vocabularies. They may be under the same namespace
        (`tokens`) or the target tokens can have a different namespace, in which case it needs to
        be specified as `target_namespace`.
    decoder_net : `DecoderNet`, required
        Module that contains implementation of neural network for decoding output elements
    max_decoding_steps : `int`, required
        Maximum length of decoded sequences.
    target_embedder : `Embedding`
        Embedder for target tokens.
    target_namespace : `str`, optional (default = `'tokens'`)
        If the target side vocabulary is different from the source side's, you need to specify the
        target's namespace here. If not, we'll assume it is "tokens", which is also the default
        choice for the source side, and this might cause them to share vocabularies.
    beam_size : `int`, optional (default = `4`)
        Width of the beam for beam search.
    tensor_based_metric : `Metric`, optional (default = `None`)
        A metric to track on validation data that takes raw tensors when its called.
        This metric must accept two arguments when called: a batched tensor
        of predicted token indices, and a batched tensor of gold token indices.
    token_based_metric : `Metric`, optional (default = `None`)
        A metric to track on validation data that takes lists of lists of tokens
        as input. This metric must accept two arguments when called, both
        of type `List[List[str]]`. The first is a predicted sequence for each item
        in the batch and the second is a gold sequence for each item in the batch.
    scheduled_sampling_ratio : `float` optional (default = `0.0`)
        Defines ratio between teacher forced training and real output usage. If its zero
        (teacher forcing only) and `decoder_net`supports parallel decoding, we get the output
        predictions in a single forward pass of the `decoder_net`.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        decoder_net: DecoderNet,
        max_decoding_steps: int,
        target_embedder: Embedding,
        target_namespace: str = "tokens",
        tie_output_embedding: bool = False,
        scheduled_sampling_ratio: float = 0,
        label_smoothing_ratio: Optional[float] = None,
        beam_size: int = 4,
        tensor_based_metric: Metric = None,
        token_based_metric: Metric = None,
        decoder_lin_emb: bool = False
    ) -> None:
        super().__init__(target_embedder)

        self._vocab = vocab
        self.beam_size = beam_size
        self.max_steps = max_decoding_steps
        # Decodes the sequence of encoded hidden states into e new sequence of hidden states.
        self._decoder_net = decoder_net
        self._max_decoding_steps = max_decoding_steps
        self._target_namespace = target_namespace
        self._label_smoothing_ratio = label_smoothing_ratio

        # At prediction time, we use a beam search to find the most likely sequence of target tokens.
        # We need the start symbol to provide as the input at the first timestep of decoding, and
        # end symbol as a way to indicate the end of the decoded sequence.
        self._start_index = self._vocab.get_token_index(START_SYMBOL, self._target_namespace)
        self._end_index = self._vocab.get_token_index(END_SYMBOL, self._target_namespace)
        self._beam_search = BeamSearch(
            self._end_index, max_steps=max_decoding_steps, beam_size=beam_size
        )

        target_vocab_size = self._vocab.get_vocab_size(self._target_namespace)

        if self.target_embedder.get_output_dim() != self._decoder_net.target_embedding_dim:
            raise ConfigurationError(
                "Target Embedder output_dim doesn't match decoder module's input."
            )

        # We project the hidden state from the decoder into the output vocabulary space
        # in order to get log probabilities of each target token, at each time step.
        self._output_projection_layer = Linear(
            self._decoder_net.get_output_dim(), target_vocab_size
        )

        if tie_output_embedding:
            if self._output_projection_layer.weight.shape != self.target_embedder.weight.shape:
                raise ConfigurationError(
                    "Can't tie embeddings with output linear layer, due to shape mismatch"
                )
            self._output_projection_layer.weight = self.target_embedder.weight

        # These metrics will be updated during training and validation
        self._tensor_based_metric = tensor_based_metric
        self._token_based_metric = token_based_metric

        self._scheduled_sampling_ratio = scheduled_sampling_ratio
        self.num_virtual_models = num_virtual_models

        self.orthogonal_embedding_emb = torch.nn.init.orthogonal_(torch.empty(self.num_virtual_models, target_embedder.get_output_dim(),requires_grad=False)).float()
        
        self.index_dict = {"[pseudo1]":0, "[pseudo2]":1, "[pseudo3]":2, "[pseudo4]":3, "[pseudo5]":4, "[pseudo6]":5, "[pseudo7]":6, "[pseudo8]":7, "[pseudo9]":8}
        self.vocab = vocab
        
        self.decoder_lin_emb = decoder_lin_emb

    def _forward_beam_search(self, state: Dict[str, torch.Tensor], index = None) -> Dict[str, torch.Tensor]:
        """
        Prepare inputs for the beam search, does beam search and returns beam search results.
        """
        batch_size = 1
        states = []
        for i in range(num_virtual_models):
            v_model = {}
            for k, tensor in state.items():
                v_model[k] = tensor[i,:].unsqueeze(0)
            states.append(v_model)

        start_predictions = [state["source_mask"].new_full((batch_size,), fill_value=self._start_index, dtype=torch.long) for state in states]

        # start_predictions = state["source_mask"].new_full(
        #     (batch_size,), fill_value=self._start_index, dtype=torch.long
        # )


        from inspect import signature

        predictions: List[torch.Tensor] = []
        backpointers: List[torch.Tensor] = []

        #配列のstart_predictions, encoder outs仕様にする
        # start_state = state
        start_state = states

        #--option1
        # start_class_log_probabilities, states = self.take_step(start_predictions, start_state)

        #--option2
        # class_log_probabilities, states = self._prepare_output_projections(start_predictions, start_state)
        # class_log_probabilities = F.softmax(class_log_probabilities, dim=-1)
        # start_class_log_probabilities = torch.log(sum(class_log_probabilities)/num_virtual_models).unsqueeze(0)

        #--option3
        new_output_projections = []
        new_states = []

        for i, (last_prediction, state) in enumerate(zip(start_predictions, states)):
            output_projections, state = self._prepare_output_projections(last_prediction, state, index = (index[i] if index is not None else index))
            new_output_projections.append(output_projections)
            new_states.append(state)

        # shape: (group_size, num_classes)
        class_log_probabilities = [F.softmax(new_output, dim=-1) for new_output in new_output_projections]
        class_log_probabilities = torch.log(sum(class_log_probabilities)/num_virtual_models)
        start_class_log_probabilities, states = class_log_probabilities, new_states

        # print("start_class_log_probabilities",start_class_log_probabilities.shape)

        num_classes = start_class_log_probabilities.size()[1]

        start_top_log_probabilities, start_predicted_classes = start_class_log_probabilities.topk(self.beam_size)

        last_log_probabilities = start_top_log_probabilities

        #predictionsは最終的に普通の配列で良い
        predictions.append(start_predicted_classes)

        log_probs_after_end = start_class_log_probabilities.new_full(
            (batch_size * self.beam_size, num_classes), float("-inf")
        )
        log_probs_after_end[:, self._end_index] = 0.0

        for state in states:
            for key, state_tensor in state.items():
                if state_tensor is None:
                    continue
                else:
                    # shape: (batch_size * beam_size, *)
                    _, *last_dims = state_tensor.size()
                    
                    state[key] = (
                        state_tensor.unsqueeze(1)
                        .expand(batch_size, self.beam_size, *last_dims)
                        .reshape(batch_size * self.beam_size, *last_dims))
        
        # encoder_outs = state
        encoder_outs = states

        for timestep in range(self.max_steps - 1):
            last_predictions = predictions[-1].reshape(batch_size * self.beam_size)

            if (last_predictions == self._end_index).all():
                break
            
            #ensembleのために確率の平均を出す
            # class_log_probabilities = self.take_step(last_predictions, state)
            # print("平均の確率になった")



            ###------------------batch version
            # last_predictions = last_predictions.repeat(num_virtual_models)

            # output_projections, state = self._prepare_output_projections(last_predictions, state)
            # class_log_probabilities = F.softmax(output_projections, dim=-1)

            # exit()
            
            # aggregated_probs = None
            # for j in range(self.beam_size):
            #         ag_probs_for_beam = torch.log(sum([class_log_probabilities[(self.beam_size*i+j),:] for i in range(num_virtual_models)])/num_virtual_models).unsqueeze(0)
            #         if aggregated_probs is None:
            #             aggregated_probs = ag_probs_for_beam
            #         else:
            #             aggregated_probs = torch.cat((aggregated_probs, ag_probs_for_beam), 0)
            
            # print("aggregated probs", aggregated_probs.shape)
            # class_log_probabilities = aggregated_probs

            # encoder_outs = state
            ##------------------
            next_probs_group = []
            new_states_group = []

            for state in encoder_outs:
                next_probs, new_state = self._prepare_output_projections(last_predictions, state)
                next_probs = F.softmax(next_probs, dim=-1)

                next_probs_group.append(next_probs)
                new_states_group.append(new_state)

            encoder_outs = new_states_group


            last_predictions_expanded = last_predictions.unsqueeze(-1).expand(
                batch_size * self.beam_size, num_classes
            )

            #ensembleのために確率の平均を出す
            class_log_probabilities = torch.log(sum(next_probs_group)/num_virtual_models)

            cleaned_log_probabilities = torch.where(
                last_predictions_expanded == self._end_index,
                log_probs_after_end,
                class_log_probabilities,)

            top_log_probabilities, predicted_classes = cleaned_log_probabilities.topk(
                self.beam_size
            )
            expanded_last_log_probabilities = (
                last_log_probabilities.unsqueeze(2)
                .expand(batch_size, self.beam_size, self.beam_size)
                .reshape(batch_size * self.beam_size, self.beam_size)
            )

            # shape: (batch_size * beam_size, per_node_beam_size)
            summed_top_log_probabilities = top_log_probabilities + expanded_last_log_probabilities

            # shape: (batch_size, beam_size * per_node_beam_size)
            reshaped_summed = summed_top_log_probabilities.reshape(
                batch_size, self.beam_size * self.beam_size
            )

            # shape: (batch_size, beam_size * per_node_beam_size)
            reshaped_predicted_classes = predicted_classes.reshape(
                batch_size, self.beam_size * self.beam_size
            )

            restricted_beam_log_probs, restricted_beam_indices = reshaped_summed.topk(
                self.beam_size
            )

            restricted_predicted_classes = reshaped_predicted_classes.gather(
                1, restricted_beam_indices
            )

            predictions.append(restricted_predicted_classes)

            last_log_probabilities = restricted_beam_log_probs

            backpointer = restricted_beam_indices // self.beam_size

            backpointers.append(backpointer)
            
            for state in encoder_outs:
                for key, state_tensor in state.items():
                    if state_tensor is None:
                        continue
                    else:
                        _, *last_dims = state_tensor.size()
                        # shape: (batch_size, beam_size, *)
                        expanded_backpointer = backpointer.view(
                            batch_size, self.beam_size, *([1] * len(last_dims))
                        ).expand(batch_size, self.beam_size, *last_dims)

                        # shape: (batch_size * beam_size, *)
                        state[key] = (
                            state_tensor.reshape(batch_size, self.beam_size, *last_dims)
                            .gather(1, expanded_backpointer)
                            .reshape(batch_size * self.beam_size, *last_dims)
                        )

        if not torch.isfinite(last_log_probabilities).all():
            print(
                "Infinite log probabilities encountered. Some final sequences may not make sense. "
                "This can happen when the beam size is larger than the number of valid (non-zero "
                "probability) transitions that the step function produces.",
            )

        reconstructed_predictions = reconstruct_sequences(predictions, backpointers)

        # shape: (batch_size, beam_size, max_steps)
        all_predictions = torch.cat(list(reversed(reconstructed_predictions)), 2)

        all_top_k_predictions = all_predictions
        last_log_probabilities = last_log_probabilities

        output_dict = {
            "class_log_probabilities": last_log_probabilities,
            "predictions": all_top_k_predictions,
        }
        

        
        return output_dict

    def _forward_loss(
        self, state: Dict[str, torch.Tensor], target_tokens: TextFieldTensors
    ) -> Dict[str, torch.Tensor]:
        """
        Make forward pass during training or do greedy search during prediction.

        Notes
        -----
        We really only use the predictions from the method to test that beam search
        with a beam size of 1 gives the same results.
        """
        # shape: (batch_size, max_input_sequence_length, encoder_output_dim)
        encoder_outputs = state["encoder_outputs"]

        # shape: (batch_size, max_input_sequence_length)
        source_mask = state["source_mask"]

        # shape: (batch_size, max_target_sequence_length)
        targets = util.get_token_ids_from_text_field_tensors(target_tokens)



        # Prepare embeddings for targets. They will be used as gold embeddings during decoder training
        # shape: (batch_size, max_target_sequence_length, embedding_dim)
        target_embedding = self.target_embedder(targets)

        # shape: (batch_size, max_target_batch_sequence_length)
        target_mask = util.get_text_field_mask(target_tokens)

        if self._scheduled_sampling_ratio == 0 and self._decoder_net.decodes_parallel:
            _, decoder_output = self._decoder_net(
                previous_state=state,
                previous_steps_predictions=target_embedding[:, :-1, :],
                encoder_outputs=encoder_outputs,
                source_mask=source_mask,
                previous_steps_mask=target_mask[:, :-1],
            )

            # shape: (group_size, max_target_sequence_length, num_classes)
            logits = self._output_projection_layer(decoder_output)
        else:
            batch_size = source_mask.size()[0]
            _, target_sequence_length = targets.size()

            # The last input from the target is either padding or the end symbol.
            # Either way, we don't have to process it.
            num_decoding_steps = target_sequence_length - 1

            # Initialize target predictions with the start index.
            # shape: (batch_size,)
            last_predictions = source_mask.new_full(
                (batch_size,), fill_value=self._start_index, dtype=torch.long
            )

            # shape: (steps, batch_size, target_embedding_dim)
            steps_embeddings = torch.Tensor([])

            step_logits: List[torch.Tensor] = []


            for timestep in range(num_decoding_steps):
                if self.training and torch.rand(1).item() < self._scheduled_sampling_ratio:
                    # Use gold tokens at test time and at a rate of 1 - _scheduled_sampling_ratio
                    # during training.
                    # shape: (batch_size, steps, target_embedding_dim)
                    state["previous_steps_predictions"] = steps_embeddings

                    # shape: (batch_size, )
                    effective_last_prediction = last_predictions
                else:
                    # shape: (batch_size, )
                    effective_last_prediction = targets[:, timestep]

                    if timestep == 0:
                        state["previous_steps_predictions"] = torch.Tensor([])
                    else:
                        # shape: (batch_size, steps, target_embedding_dim)
                        state["previous_steps_predictions"] = target_embedding[:, :timestep]

                # shape: (batch_size, num_classes)
                output_projections, state = self._prepare_output_projections(
                    effective_last_prediction, state
                )

                # list of tensors, shape: (batch_size, 1, num_classes)
                step_logits.append(output_projections.unsqueeze(1))

                # shape (predicted_classes): (batch_size,)
                _, predicted_classes = torch.max(output_projections, 1)

                # shape (predicted_classes): (batch_size,)
                last_predictions = predicted_classes

                # shape: (batch_size, 1, target_embedding_dim)
                last_predictions_embeddings = self.target_embedder(last_predictions).unsqueeze(1)

                # This step is required, since we want to keep up two different prediction history: gold and real
                if steps_embeddings.shape[-1] == 0:
                    # There is no previous steps, except for start vectors in `last_predictions`
                    # shape: (group_size, 1, target_embedding_dim)
                    steps_embeddings = last_predictions_embeddings
                else:
                    # shape: (group_size, steps_count, target_embedding_dim)
                    steps_embeddings = torch.cat([steps_embeddings, last_predictions_embeddings], 1)

            # shape: (batch_size, num_decoding_steps, num_classes)
            logits = torch.cat(step_logits, 1)

        # Compute loss.

        target_mask = util.get_text_field_mask(target_tokens)
        loss = self._get_loss(logits, targets, target_mask)


        # TODO: We will be using beam search to get predictions for validation, but if beam size in 1
        # we could consider taking the last_predictions here and building step_predictions
        # and use that instead of running beam search again, if performance in validation is taking a hit
        output_dict = {"loss": loss}

        return output_dict

    def _prepare_output_projections(
        self, last_predictions: torch.Tensor, state: Dict[str, torch.Tensor], index = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Decode current state and last prediction to produce produce projections
        into the target space, which can then be used to get probabilities of
        each target token for the next step.

        Inputs are the same as for `take_step()`.
        """
        # shape: (group_size, max_input_sequence_length, encoder_output_dim)
        encoder_outputs = state["encoder_outputs"]


        # shape: (group_size, max_input_sequence_length)
        source_mask = state["source_mask"]


        # shape: (group_size, steps_count, decoder_output_dim)
        previous_steps_predictions = state.get("previous_steps_predictions")


        # shape: (batch_size, 1, target_embedding_dim)
        # last_predictions = last_predictions.unsqueeze(0)
        # last_predictions = last_predictions.unsqueeze(0).expand(self.beam_size*9, -1).reshape(1,-1)
        last_predictions_embeddings = self.target_embedder(last_predictions).unsqueeze(1)

        if index is not None:
            orthogonal_vecs = torch.index_select(self.orthogonal_embedding_emb.to(last_predictions_embeddings.device),0,index).unsqueeze(1)
            bsz, seq_len, emb_dim = last_predictions_embeddings.size()
            orthogonal_vecs = orthogonal_vecs.expand(-1, seq_len, -1)
            
            scale = 1

            last_predictions_embeddings += (scale*orthogonal_vecs)



        if previous_steps_predictions is None or previous_steps_predictions.shape[-1] == 0:
            # There is no previous steps, except for start vectors in `last_predictions`
            # shape: (group_size, 1, target_embedding_dim)
            # exit()
            previous_steps_predictions = last_predictions_embeddings

            
        else:
            # shape: (group_size, steps_count, target_embedding_dim)
            previous_steps_predictions = torch.cat(
                [previous_steps_predictions, last_predictions_embeddings], 1
            )

        decoder_state, decoder_output = self._decoder_net(
            previous_state=state,
            encoder_outputs=encoder_outputs,
            source_mask=source_mask,
            previous_steps_predictions=previous_steps_predictions,
        )
        state["previous_steps_predictions"] = previous_steps_predictions

        # Update state with new decoder state, override previous state
        state.update(decoder_state)

        if self._decoder_net.decodes_parallel:
            decoder_output = decoder_output[:, -1, :]

        # shape: (group_size, num_classes)
        output_projections = self._output_projection_layer(decoder_output)

        return output_projections, state

    def _get_loss(
        self, logits: torch.LongTensor, targets: torch.LongTensor, target_mask: torch.BoolTensor
    ) -> torch.Tensor:
        """
        Compute loss.

        Takes logits (unnormalized outputs from the decoder) of size (batch_size,
        num_decoding_steps, num_classes), target indices of size (batch_size, num_decoding_steps+1)
        and corresponding masks of size (batch_size, num_decoding_steps+1) steps and computes cross
        entropy loss while taking the mask into account.

        The length of `targets` is expected to be greater than that of `logits` because the
        decoder does not need to compute the output corresponding to the last timestep of
        `targets`. This method aligns the inputs appropriately to compute the loss.

        During training, we want the logit corresponding to timestep i to be similar to the target
        token from timestep i + 1. That is, the targets should be shifted by one timestep for
        appropriate comparison.  Consider a single example where the target has 3 words, and
        padding is to 7 tokens.
           The complete sequence would correspond to <S> w1  w2  w3  <E> <P> <P>
           and the mask would be                     1   1   1   1   1   0   0
           and let the logits be                     l1  l2  l3  l4  l5  l6
        We actually need to compare:
           the sequence           w1  w2  w3  <E> <P> <P>
           with masks             1   1   1   1   0   0
           against                l1  l2  l3  l4  l5  l6
           (where the input was)  <S> w1  w2  w3  <E> <P>
        """
        # shape: (batch_size, num_decoding_steps)
        relevant_targets = targets[:, 1:].contiguous()

        # shape: (batch_size, num_decoding_steps)
        relevant_mask = target_mask[:, 1:].contiguous()

        return util.sequence_cross_entropy_with_logits(
            logits, relevant_targets, relevant_mask, label_smoothing=self._label_smoothing_ratio
        )

    def get_output_dim(self):
        return self._decoder_net.get_output_dim()

    def take_step(
        self, last_predictions: torch.Tensor, state: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        output_projections, state = self._prepare_output_projections(last_predictions, state)

        class_log_probabilities = F.softmax(output_projections, dim=-1)
        class_log_probabilities = torch.log(sum(class_log_probabilities)/num_virtual_models).unsqueeze(0)

        return class_log_probabilities, state


    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        all_metrics: Dict[str, float] = {}
        if not self.training:
            if self._tensor_based_metric is not None:
                all_metrics.update(
                    self._tensor_based_metric.get_metric(reset=reset)  # type: ignore
                )
            if self._token_based_metric is not None:
                all_metrics.update(self._token_based_metric.get_metric(reset=reset))  # type: ignore
        return all_metrics

    @overrides
    def forward(
        self,
        encoder_out: Dict[str, torch.LongTensor],
        target_tokens: TextFieldTensors = None,
        source_tokens: TextFieldTensors = None
    ) -> Dict[str, torch.Tensor]:
        state = encoder_out
        decoder_init_state = self._decoder_net.init_decoder_state(state)
        state.update(decoder_init_state)


        if self.decoder_lin_emb:
            try:
                index = torch.tensor([self.index_dict[self.vocab.get_token_from_index(i, namespace = "source_tokens")] for \
                    i in source_tokens["source_tokens"]["tokens"][:,1].squeeze(0).tolist()]).long().to(source_tokens["source_tokens"]["tokens"].device)
            except:
                import sys
                sys.stdout.write("AAA {}".format(sys.exc_info()))
                index = torch.tensor([self.index_dict[self.vocab.get_token_from_index(i,  namespace = "source_tokens")] for \
                    i in source_tokens["source_tokens"]["tokens"][:,1].squeeze(0).tolist()]).long().to(source_tokens["source_tokens"]["tokens"].device)
        


        if target_tokens:
            state_forward_loss = (
                state if self.training else {k: v.clone() for k, v in state.items()}
            )
            output_dict = self._forward_loss(state_forward_loss, target_tokens)
        else:
            output_dict = {}

        if not self.training:
            #完成（出力の確率を平均化）
            if self.decoder_lin_emb:
                predictions = self._forward_beam_search(state, index)
            else:
                predictions = self._forward_beam_search(state)
            output_dict.update(predictions)

            #target tokensを1つにする
            

            if target_tokens:

                target_tokens["target_tokens"]["tokens"] = target_tokens["target_tokens"]["tokens"][0,:].unsqueeze(0) #target tokensを1つにした
                targets = util.get_token_ids_from_text_field_tensors(target_tokens)
                if self._tensor_based_metric is not None:
                    # shape: (batch_size, beam_size, max_sequence_length)
                    top_k_predictions = output_dict["predictions"]
                    # shape: (batch_size, max_predicted_sequence_length)
                    best_predictions = top_k_predictions[:, 0, :]

                    self._tensor_based_metric(best_predictions, targets)  # type: ignore
                if self._token_based_metric is not None:
                    output_dict = self.post_process(output_dict)
                    predicted_tokens = output_dict["predicted_tokens"]

                    self._token_based_metric(  # type: ignore
                        predicted_tokens,
                        self.indices_to_tokens(targets[:, 1:]),
                    )
        return output_dict

    @overrides
    def post_process(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        This method trims the output predictions to the first end symbol, replaces indices with
        corresponding tokens, and adds a field called `predicted_tokens` to the `output_dict`.
        """
        predicted_indices = output_dict["predictions"]
        all_predicted_tokens = self.indices_to_tokens(predicted_indices)
        output_dict["predicted_tokens"] = all_predicted_tokens
        return output_dict

    def indices_to_tokens(self, batch_indeces: numpy.ndarray) -> List[List[str]]:

        if not isinstance(batch_indeces, numpy.ndarray):
            batch_indeces = batch_indeces.detach().cpu().numpy()

        all_tokens = []
        for indices in batch_indeces:
            # Beam search gives us the top k results for each source sentence in the batch
            # but we just want the single best.
            if len(indices.shape) > 1:
                indices = indices[0]
            indices = list(indices)
            # Collect indices till the first end_symbol
            if self._end_index in indices:
                indices = indices[: indices.index(self._end_index)]
            tokens = [
                self._vocab.get_token_from_index(x, namespace=self._target_namespace)
                for x in indices
            ]
            all_tokens.append(tokens)

        return all_tokens
