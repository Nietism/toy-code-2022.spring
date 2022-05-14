import copy
import math
import random
from typing import *

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from transformers import modeling_bart as bart
from transformers.modeling_utils import BeamHypotheses, calc_banned_ngram_tokens, calc_banned_bad_words_ids, \
    top_k_top_p_filtering

entity_dict = {'VEH': 'Vehicle', 'PER': 'Person', 'LOC': 'Location', 'Job-Title': 'Job-Title',
               'ORG': 'Organization',
               'GPE': 'Geopolitical-Entity', 'Time': 'Time', 'FAC': 'Facility', 'Numeric': 'Numeric',
               'WEA': 'Weapons', 'TIM': 'Time',
               'Sentence': 'Sentence', 'Crime': 'Crime', 'Contact-Info': 'Contact-Information'}


def extract_backreferences(ids, num_embeddings, backpointer_idx):
    ids_mask = ids >= num_embeddings
    backreferences = ids.clone() - num_embeddings
    backreferences[~ids_mask] = 0
    backreferences += (~ids_mask).long() * torch.arange(
        ids.size(1),
        dtype=ids.dtype,
        device=ids.device)
    ids = ids.clone()
    ids[ids_mask] = backpointer_idx
    return ids, backreferences


class ACEBartEncoder(nn.Module):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer
    is a :class:`EncoderLayer`.

    Args:
        config: BartConfig
    """

    def __init__(self, config: bart.BartConfig, embed_tokens, backpointer_idx):
        super().__init__()

        self.backpointer_idx = backpointer_idx

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states

        embed_dim = embed_tokens.embedding_dim
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = config.max_position_embeddings

        self.embed_tokens = embed_tokens

        if config.static_position_embeddings:
            self.embed_positions = bart.SinusoidalPositionalEmbedding(
                config.max_position_embeddings, embed_dim, self.padding_idx
            )
        else:
            self.embed_positions = bart.LearnedPositionalEmbedding(
                config.max_position_embeddings, embed_dim, self.padding_idx,  # config.extra_pos_embeddings,
            )

        self.layers = nn.ModuleList([bart.EncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layernorm_embedding = bart.LayerNorm(embed_dim) if config.normalize_embedding else nn.Identity()
        # mbart has one extra layer_norm
        self.layer_norm = bart.LayerNorm(config.d_model) if config.normalize_before else None

    def forward(
            self, input_ids, embedded=None, attention_mask=None,
    ):
        """
        Args:
            input_ids (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            attention_mask (torch.LongTensor): indicating which indices are padding tokens.
        Returns:
            Tuple comprised of:
                - **x** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *self.output_hidden_states:* is True.
                - **all_attentions** (List[Tensor]): Attention weights for each layer.
                During training might not be of length n_layers because of layer dropout.
        """
        # check attention mask and invert
        if attention_mask is not None:
            attention_mask = bart.invert_mask(attention_mask)

        input_ids, backreferences = extract_backreferences(
            input_ids, self.embed_tokens.num_embeddings, self.backpointer_idx)
        inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
        embed_pos = self.embed_positions(input_ids)
        x = inputs_embeds + embed_pos

        if embedded is not None:
            x += embedded

        x = self.layernorm_embedding(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_states, all_attentions = [], []
        for encoder_layer in self.layers:
            if self.output_hidden_states:
                encoder_states.append(x)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                attn = None
            else:
                x, attn = encoder_layer(x, attention_mask)

            if self.output_attentions:
                all_attentions.append(attn)

        if self.layer_norm:
            x = self.layer_norm(x)
        if self.output_hidden_states:
            encoder_states.append(x)

        # T x B x C -> B x T x C
        encoder_states = [hidden_state.transpose(0, 1) for hidden_state in encoder_states]
        x = x.transpose(0, 1)

        return x, encoder_states, all_attentions


class ACEBartDecoder(nn.Module):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer
    is a :class:`DecoderLayer`.
    Args:
        config: BartConfig
        embed_tokens (torch.nn.Embedding): output embedding
    """

    def __init__(self, config: bart.BartConfig, embed_tokens: nn.Embedding, backpointer_idx, amr_mode=True):
        super().__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0
        self.backpointer_idx = backpointer_idx

        embed_dim = embed_tokens.embedding_dim

        self.embed_tokens = embed_tokens
        if config.static_position_embeddings:
            self.embed_positions = bart.SinusoidalPositionalEmbedding(
                config.max_position_embeddings, embed_dim, self.padding_idx
            )
        else:
            self.embed_positions = bart.LearnedPositionalEmbedding(
                config.max_position_embeddings, embed_dim, self.padding_idx,  # config.extra_pos_embeddings,
            )

        self.layers = nn.ModuleList(
            [bart.DecoderLayer(config) for _ in range(config.decoder_layers)]
        )  # type: List[DecoderLayer]
        self.layernorm_embedding = bart.LayerNorm(config.d_model) if config.normalize_embedding else nn.Identity()
        self.layer_norm = bart.LayerNorm(config.d_model) if config.add_final_layer_norm else None

        self.pointer_k = nn.Linear(config.d_model, config.d_model)
        # self.pointer_k.weight.data = self.layers[-1].self_attn.k_proj.weight.data.clone()

        self.pointer_q = nn.Linear(config.d_model, config.d_model)
        # self.pointer_q.weight.data = self.layers[-1].self_attn.q_proj.weight.data.clone()

        # self.pointer_k = nn.Sequential(
        #     nn.Linear(config.d_model, config.decoder_ffn_dim),
        #     nn.GELU(),
        #     nn.Linear(config.decoder_ffn_dim, config.d_model),
        # )
        # self.pointer_q = nn.Sequential(
        #     nn.Linear(config.d_model, config.decoder_ffn_dim),
        #     nn.GELU(),
        #     nn.Linear(config.decoder_ffn_dim, config.d_model),
        # )

        self.amr_mode = amr_mode

    def forward(
            self,
            input_ids,
            encoder_hidden_states,
            encoder_padding_mask,
            decoder_padding_mask,
            decoder_causal_mask,
            decoder_cached_states=None,
            use_cache=False,
            **unused
    ):
        """
        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            input_ids (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_hidden_states: output from the encoder, used for
                encoder-side attention
            encoder_padding_mask: for ignoring pad tokens
            decoder_cached_states (dict or None): dictionary used for storing state during generation

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - hidden states
                - attentions
        """

        # check attention mask and invert
        if encoder_padding_mask is not None:
            encoder_padding_mask = bart.invert_mask(encoder_padding_mask)

        input_ids, backreferences = extract_backreferences(
            input_ids,
            self.embed_tokens.num_embeddings,
            self.backpointer_idx)
        # embed positions
        embed_pos = self.embed_positions(input_ids, use_cache=use_cache)
        positions = embed_pos

        # to do this during prediction the old positions should be removed
        if use_cache:
            input_ids = input_ids[:, -1:]
            positions = positions[:, -1:]  # happens after we embed them
            # assert input_ids.ne(self.padding_idx).any()

        x = self.embed_tokens(input_ids) * self.embed_scale
        x += positions
        x = self.layernorm_embedding(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Convert to Bart output format: (seq_len, BS, model_dim) -> (BS, seq_len, model_dim)
        x = x.transpose(0, 1)
        encoder_hidden_states = encoder_hidden_states.transpose(0, 1)

        # decoder layers
        all_hidden_states = ()
        all_self_attns = ()
        next_decoder_cache = []
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if self.output_hidden_states:
                all_hidden_states += (x,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            layer_state = decoder_cached_states[idx] if decoder_cached_states is not None else None

            x, layer_self_attn, layer_past = decoder_layer(
                x,
                encoder_hidden_states,
                encoder_attn_mask=encoder_padding_mask,
                decoder_padding_mask=decoder_padding_mask,
                layer_state=layer_state,
                causal_mask=decoder_causal_mask,
            )

            if use_cache:
                next_decoder_cache.append(layer_past.copy())

            if self.layer_norm and (idx == len(self.layers) - 1):  # last layer of mbart
                x = self.layer_norm(x)
            if self.output_attentions:
                all_self_attns += (layer_self_attn,)

        # Convert to standard output format: (seq_len, BS, model_dim) -> (BS, seq_len, model_dim)
        all_hidden_states = [hidden_state.transpose(0, 1) for hidden_state in all_hidden_states]
        x = x.transpose(0, 1)
        encoder_hidden_states = encoder_hidden_states.transpose(0, 1)

        xq = self.pointer_q(x)
        xk = self.pointer_k(x)

        if decoder_cached_states is not None:
            if 'prev_key' in decoder_cached_states[-1].get('pointer', {}):
                last_state = decoder_cached_states[-1]['pointer']
                xk = torch.cat([last_state['prev_key'], xk], dim=1)

        next_state = {'pointer': {'prev_key': xk}}

        if use_cache:
            next_decoder_cache.append(next_state)

        if self.amr_mode:
            scores = torch.einsum('bqh,bkh->bqk', xq, xk)

            if decoder_cached_states:
                mask = torch.full_like(scores[0], float('-inf'))
                mask = mask.triu(diagonal=xk.size(1) - 1)
            else:
                mask = torch.full_like(scores[0], float('-inf'))
                mask = mask.triu()
            scores += mask.unsqueeze(0)
        else:
            scores = torch.full((xq.size(0), xq.size(1), xk.size(1)), float('-inf'), device=xq.device)

        if use_cache:
            next_cache = ((encoder_hidden_states, encoder_padding_mask), next_decoder_cache)
        else:
            next_cache = None
        return (x, scores), next_cache, all_hidden_states, list(all_self_attns)


class ACEBartModel(bart.PretrainedBartModel):
    def __init__(self, config: bart.BartConfig, backpointer_idx=None):
        super().__init__(config)
        self.output_attentions = True
        self.output_hidden_states = config.output_hidden_states

        self.padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, self.padding_idx)

        if backpointer_idx is not None:
            self.backpointer_idx = backpointer_idx
        else:
            self.backpointer_idx = self.shared.num_embeddings - 1

        self.encoder = ACEBartEncoder(config, self.shared, backpointer_idx=self.backpointer_idx)
        self.decoder = ACEBartDecoder(config, self.shared, backpointer_idx=self.backpointer_idx)

        self.init_weights()

    @property
    def sentence_mode(self):
        return self.decoder.amr_mode

    @sentence_mode.setter
    def sentence_mode(self, value):
        assert isinstance(value, bool)
        self.decoder.amr_mode = value

    def forward(
            self,
            input_ids,
            attention_mask=None,
            decoder_input_ids=None,
            encoder_outputs: Optional[Tuple] = None,
            decoder_attention_mask=None,
            decoder_cached_states=None,
            use_cache=False,
    ):

        # make masks if user doesn't supply
        if not use_cache:
            decoder_input_ids, decoder_padding_mask, causal_mask = bart._prepare_bart_decoder_inputs(
                self.config,
                input_ids,
                decoder_input_ids=decoder_input_ids,
                decoder_padding_mask=decoder_attention_mask,
                causal_mask_dtype=self.shared.weight.dtype,
            )
        else:
            decoder_padding_mask, causal_mask = None, None

        assert decoder_input_ids is not None
        if encoder_outputs is None:
            encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        assert isinstance(encoder_outputs, tuple)
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            decoder_input_ids,
            encoder_outputs[0],
            attention_mask,
            decoder_padding_mask,
            decoder_causal_mask=causal_mask,
            decoder_cached_states=decoder_cached_states,
            use_cache=use_cache,
        )
        # Attention and hidden_states will be [] or None if they aren't needed
        # decoder_outputs: Tuple = bart._filter_out_falsey_values(decoder_outputs)
        assert isinstance(decoder_outputs[0][0], torch.Tensor)
        assert isinstance(decoder_outputs[0][1], torch.Tensor)
        encoder_outputs: Tuple = bart._filter_out_falsey_values(encoder_outputs)
        return decoder_outputs + encoder_outputs

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def get_output_embeddings(self):
        return bart._make_linear_from_emb(self.shared)  # make it on the fly


class ACEBartForConditionalGeneration(bart.PretrainedBartModel):
    base_model_prefix = "model"

    def __init__(self, config: bart.BartConfig, backpointer_idx=None):
        super().__init__(config)
        base_model = ACEBartModel(config, backpointer_idx)
        self.model = base_model
        self.pad_index = base_model.shared.padding_idx
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.backpointer_idx = backpointer_idx
        self._rev = None

    def init_reverse_model(self):
        rev = ACEBartForConditionalGeneration(self.model.config, self.backpointer_idx)
        rev.model.shared = self.model.shared
        rev.model.encoder = self.model.encoder
        rev.model.decoder.embed_tokens = self.model.decoder.embed_tokens
        rev.model.decoder.embed_positions = self.model.decoder.embed_positions
        self.amr_mode = True
        rev.amr_mode = False
        self._rev = rev

    @property
    def rev(self):
        if self._rev is None:
            return self
        else:
            return self._rev

    @property
    def amr_mode(self):
        return self.model.decoder.amr_mode

    @amr_mode.setter
    def amr_mode(self, value):
        assert isinstance(value, bool)
        self.model.decoder.amr_mode = value

    def prepare_inputs_for_predict(self, input_ids, **kwargs):
        return {"input_ids": input_ids}

    def get_event_ids(self, tokenizer, e_d_list):
        e_d_id_list = []
        for e_d in e_d_list:
            g_event = tokenizer.INIT + e_d
            if g_event in tokenizer.encoder:
                g_event_id = tokenizer.encoder[g_event]
                e_d_id_list.append(g_event_id)
        return e_d_id_list

    def get_special_ids(self, tokenizer, y, entitys):
        # pass
        event_ids = []
        for event in tokenizer.event_types:
            g_event = tokenizer.INIT + event
            g_event_id = tokenizer.encoder[g_event]
            event_ids.append(g_event_id)
        arg_indexes_ids = []
        for a_index in tokenizer.arg_indexes:
            arg_indexes_id = tokenizer.encoder[tokenizer.INIT + a_index]
            arg_indexes_ids.append(arg_indexes_id)
        role_ids = []
        for r in tokenizer.roles:
            g_r = tokenizer.INIT + r
            g_r_id = tokenizer.encoder[g_r]
            role_ids.append(g_r_id)
        arg_ids = []
        for a in tokenizer.args:
            g_a = tokenizer.INIT + a
            g_a_id = tokenizer.encoder[g_a]
            arg_ids.append(g_a_id)
        entity_ids = []
        for a in tokenizer.entity:
            g_a = tokenizer.INIT + a
            g_a_id = tokenizer.encoder[g_a]
            entity_ids.append(g_a_id)
        entity_arg_ids = []
        for a in tokenizer.entity_arg:
            g_a = tokenizer.INIT + a
            if g_a in tokenizer.encoder:
                g_a_id = tokenizer.encoder[g_a]
            entity_arg_ids.append(g_a_id)
        y_args_ids = []
        y_entity_ids = []
        y_token = y['decoder_input_ids'][0]
        total_args_number = 0
        arg_candidates = []
        entity_type_candidates = []
        entity_type_arg_dict = {}
        for ety in entitys:
            entity_type = ety['entity-type'].split(':')[0]
            if entity_type in entity_dict:
                entity_type = entity_dict[entity_type]
            etype_token_str = tokenizer.INIT + entity_type
            if etype_token_str not in tokenizer.encoder:
                pass
            else:
                etype_token_str_id = tokenizer.encoder[etype_token_str]
                if int(etype_token_str_id) not in entity_type_candidates:
                    entity_type_candidates.append(int(etype_token_str_id))
                if int(etype_token_str_id) not in entity_type_arg_dict:
                    entity_type_arg_dict[int(etype_token_str_id)] = []
                head = ety['head'].split()[-1]
                ety_token_str = tokenizer.INIT + head
                if ety_token_str not in tokenizer.encoder or head not in tokenizer.entity_arg:
                    pass
                else:
                    ety_token_str_id = tokenizer.encoder[ety_token_str]
                    if int(ety_token_str_id) not in arg_candidates:
                        arg_candidates.append(int(ety_token_str_id))
                    if int(ety_token_str_id) not in entity_type_arg_dict[int(etype_token_str_id)]:
                        entity_type_arg_dict[int(etype_token_str_id)].append(int(ety_token_str_id))
        # seq = []
        # for token in y_token:
        #     seq.append(tokenizer.decoder[int(token)])
        entity_arg_dict = {}
        event_type_list = []
        # 取出decoder中的argument(通过entity可以)
        for id, token in enumerate(y_token):
            if int(token) in entity_ids:
                if int(token) not in entity_arg_dict:
                    entity_arg_dict[int(token)] = []
            if int(token) in event_ids:
                event_type_list.append(int(token))

        for id, token in enumerate(y_token):  # 候选论元
            if int(token) in arg_ids:
                y_args_ids.append(int(token))
            if int(token) in entity_ids:
                y_entity_ids.append(int(token))
            if int(token) in arg_indexes_ids:
                total_args_number += 1
                if int(y_token[id + 1]) in arg_ids and int(y_token[id - 1]) in entity_ids:
                    entity_arg_dict[int(y_token[id - 1])].append(int(y_token[id + 1]))
        # seq_dict = {}
        # for key, value in entity_arg_dict.items():
        #     seq_dict[tokenizer.decoder[int(key)]] = [tokenizer.decoder[int(v)] for v in value]

        label_dict = {}
        # label_dict[root_token] = [self.vocab.get_token_index('root_label', 'head_tags')]
        label_dict[tokenizer.encoder[tokenizer.INIT + "Life:Be-Born"]] = [
            tokenizer.encoder[tokenizer.INIT + role] for role in
            ['Place', 'Person', 'Time-Within', 'Time-Holds']]
        label_dict[tokenizer.encoder[tokenizer.INIT + "Life:Marry"]] = [
            tokenizer.encoder[tokenizer.INIT + role] for role in
            ['Person', 'Time-Within', 'Place', 'Time-Holds', 'Time-Before']]
        label_dict[tokenizer.encoder[tokenizer.INIT + "Life:Injure"]] = [
            tokenizer.encoder[tokenizer.INIT + role] for role in
            ['Victim', 'Instrument', 'Agent', 'Place', 'Time-Within']]

        label_dict[tokenizer.encoder[tokenizer.INIT + "Life:Divorce"]] = [
            tokenizer.encoder[tokenizer.INIT + role] for role in ['Person', 'Time-Within']]
        label_dict[tokenizer.encoder[tokenizer.INIT + "Life:Die"]] = [
            tokenizer.encoder[tokenizer.INIT + role] for role in
            ['Victim', 'Agent', 'Place', 'Instrument', 'Time-Within', 'Time-Starting', 'Time-Holds', 'Time-Ending',
             'Person', 'Time-After', 'Time-Before', 'Time-At-Beginning']]
        label_dict[tokenizer.encoder[tokenizer.INIT + "Movement:Transport"]] = [
            tokenizer.encoder[tokenizer.INIT + role] for role in
            ['Artifact', 'Destination', 'Origin', 'Time-Before', 'Agent', 'Time-At-Beginning',
             'Time-Within', 'Vehicle', 'Time-Starting', 'Time-Ending', 'Time-After', 'Time-At-End',
             'Time-Holds', 'Victim', 'Place']]
        label_dict[tokenizer.encoder[tokenizer.INIT + "Transaction:Transfer-Ownership"]] = [
            tokenizer.encoder[tokenizer.INIT + role] for role in
            ['Artifact', 'Seller', 'Buyer', 'Place', 'Time-Within', 'Price', 'Beneficiary',
             'Time-Before', 'Time-At-Beginning', 'Time-Ending']]
        label_dict[tokenizer.encoder[tokenizer.INIT + "Transaction:Transfer-Money"]] = [
            tokenizer.encoder[tokenizer.INIT + role] for role in
            ['Recipient', 'Giver', 'Money', 'Time-Within', 'Beneficiary', 'Place', 'Time-After',
             'Time-Before', 'Time-Holds', 'Time-Starting']]
        label_dict[tokenizer.encoder[tokenizer.INIT + "Business:Start-Org"]] = [
            tokenizer.encoder[tokenizer.INIT + role] for role in
            ['Place', 'Org', 'Agent', 'Time-Within', 'Time-Before', 'Time-Starting', 'Time-After']]
        label_dict[tokenizer.encoder[tokenizer.INIT + "Business:Merge-Org"]] = [
            tokenizer.encoder[tokenizer.INIT + role] for role in ['Org', 'Time-Ending']]
        label_dict[tokenizer.encoder[tokenizer.INIT + "Business:Declare-Bankruptcy"]] = [
            tokenizer.encoder[tokenizer.INIT + role] for role in
            ['Org', 'Time-Within', 'Place', 'Time-At-Beginning', 'Time-After']]

        label_dict[tokenizer.encoder[tokenizer.INIT + "Business:End-Org"]] = [
            tokenizer.encoder[tokenizer.INIT + role] for role in
            ['Org', 'Place', 'Time-Within', 'Time-Holds', 'Time-After', 'Time-At-Beginning']]
        label_dict[tokenizer.encoder[tokenizer.INIT + "Conflict:Attack"]] = [
            tokenizer.encoder[tokenizer.INIT + role] for role in
            ['Time-Ending', 'Attacker', 'Target', 'Place', 'Instrument', 'Time-Within', 'Time-Holds',
             'Time-Before', 'Time-At-Beginning', 'Time-After', 'Time-Starting', 'Time-At-End', 'Victim',
             'Agent']]
        label_dict[tokenizer.encoder[tokenizer.INIT + "Conflict:Demonstrate"]] = [
            tokenizer.encoder[tokenizer.INIT + role] for role in
            ['Place', 'Entity', 'Time-Within', 'Time-Starting', 'Time-At-End']]
        label_dict[tokenizer.encoder[tokenizer.INIT + "Contact:Meet"]] = [
            tokenizer.encoder[tokenizer.INIT + role] for role in
            ['Entity', 'Place', 'Time-Starting', 'Time-Within', 'Time-Holds', 'Time-At-Beginning', 'Time-After',
             'Time-Ending']]
        label_dict[tokenizer.encoder[tokenizer.INIT + "Contact:Phone-Write"]] = [
            tokenizer.encoder[tokenizer.INIT + role] for role in
            ['Entity', 'Time-Starting', 'Place', 'Time-Holds', 'Time-Within', 'Time-Before',
             'Time-After']]
        label_dict[tokenizer.encoder[tokenizer.INIT + "Personnel:Start-Position"]] = [
            tokenizer.encoder[tokenizer.INIT + role] for role in
            ['Person', 'Entity', 'Position', 'Time-After', 'Time-Starting', 'Place',
             'Time-At-Beginning', 'Time-Within', 'Time-Before', 'Time-Holds']]
        label_dict[tokenizer.encoder[tokenizer.INIT + "Personnel:End-Position"]] = [
            tokenizer.encoder[tokenizer.INIT + role] for role in
            ['Person', 'Entity', 'Position', 'Time-Within', 'Place', 'Time-Before', 'Time-Starting',
             'Time-At-End', 'Time-Ending', 'Time-Holds', 'Time-After']]
        label_dict[tokenizer.encoder[tokenizer.INIT + "Personnel:Nominate"]] = [
            tokenizer.encoder[tokenizer.INIT + role] for role in
            ['Person', 'Position', 'Agent', 'Time-Within']]
        label_dict[tokenizer.encoder[tokenizer.INIT + "Personnel:Elect"]] = [
            tokenizer.encoder[tokenizer.INIT + role] for role in
            ['Person', 'Position', 'Time-Within', 'Entity', 'Place', 'Time-At-Beginning', 'Time-Starting', 'Time-Holds',
             'Time-Before']]
        label_dict[tokenizer.encoder[tokenizer.INIT + "Justice:Arrest-Jail"]] = [
            tokenizer.encoder[tokenizer.INIT + role] for role in
            ['Person', 'Agent', 'Place', 'Crime', 'Time-Within', 'Time-Holds', 'Time-Starting',
             'Time-At-Beginning', 'Time-Ending', 'Time-Before']]
        label_dict[tokenizer.encoder[tokenizer.INIT + "Justice:Release-Parole"]] = [
            tokenizer.encoder[tokenizer.INIT + role] for role in
            ['Person', 'Place', 'Time-Within', 'Entity', 'Crime', 'Time-After']]
        label_dict[tokenizer.encoder[tokenizer.INIT + "Justice:Trial-Hearing"]] = [
            tokenizer.encoder[tokenizer.INIT + role] for role in
            ['Place', 'Time-Starting', 'Time-Within', 'Time-Holds', 'Defendant', 'Crime',
             'Prosecutor', 'Adjudicator', 'Time-At-End']]
        label_dict[tokenizer.encoder[tokenizer.INIT + "Justice:Charge-Indict"]] = [
            tokenizer.encoder[tokenizer.INIT + role] for role in
            ['Defendant', 'Place', 'Prosecutor', 'Time-Within', 'Crime', 'Adjudicator', 'Time-Before', 'Time-Ending']]
        label_dict[tokenizer.encoder[tokenizer.INIT + "Justice:Sue"]] = [
            tokenizer.encoder[tokenizer.INIT + role] for role in
            ['Adjudicator', 'Defendant', 'Crime', 'Place', 'Plaintiff', 'Time-Within', 'Time-Holds']]
        label_dict[tokenizer.encoder[tokenizer.INIT + "Justice:Convict"]] = [
            tokenizer.encoder[tokenizer.INIT + role] for role in
            ['Defendant', 'Crime', 'Time-Within', 'Adjudicator', 'Place', 'Time-At-Beginning']]
        label_dict[tokenizer.encoder[tokenizer.INIT + "Justice:Sentence"]] = [
            tokenizer.encoder[tokenizer.INIT + role] for role in
            ['Defendant', 'Sentence', 'Adjudicator', 'Crime', 'Time-Within', 'Place', 'Time-At-End',
             'Time-Starting']]
        label_dict[tokenizer.encoder[tokenizer.INIT + "Justice:Fine"]] = [
            tokenizer.encoder[tokenizer.INIT + role] for role in
            ['Entity', 'Money', 'Place', 'Adjudicator', 'Time-Within', 'Crime']]
        label_dict[tokenizer.encoder[tokenizer.INIT + "Justice:Execute"]] = [
            tokenizer.encoder[tokenizer.INIT + role] for role in
            ['Agent', 'Person', 'Crime', 'Time-At-Beginning', 'Time-Within', 'Place', 'Time-After']]
        label_dict[tokenizer.encoder[tokenizer.INIT + "Justice:Extradite"]] = [
            tokenizer.encoder[tokenizer.INIT + role] for role in
            ['Person', 'Destination', 'Agent', 'Time-Within', 'Origin']]
        label_dict[tokenizer.encoder[tokenizer.INIT + "Justice:Acquit"]] = [
            tokenizer.encoder[tokenizer.INIT + role] for role in
            ['Time-Within', 'Adjudicator', 'Defendant', 'Crime']]
        label_dict[tokenizer.encoder[tokenizer.INIT + "Justice:Pardon"]] = [
            tokenizer.encoder[tokenizer.INIT + role] for role in ['Defendant', 'Adjudicator', 'Place', 'Time-At-End']]
        label_dict[tokenizer.encoder[tokenizer.INIT + "Justice:Appeal"]] = [
            tokenizer.encoder[tokenizer.INIT + role] for role in
            ['Plaintiff', 'Adjudicator', 'Place', 'Crime', 'Time-Within', 'Time-Holds']]
        role_entity_type = {}
        role_entity_type[tokenizer.encoder[tokenizer.INIT + "Person"]] = [
            tokenizer.encoder[tokenizer.INIT + entity] for entity in
            ['Person']]

        role_entity_type[tokenizer.encoder[tokenizer.INIT + "Time"]] = [
            tokenizer.encoder[tokenizer.INIT + entity] for entity in
            ['Time']]

        role_entity_type[tokenizer.encoder[tokenizer.INIT + "Place"]] = [
            tokenizer.encoder[tokenizer.INIT + entity] for entity in
            [entity_dict['GPE'], entity_dict['LOC'], entity_dict['FAC']]]

        role_entity_type[tokenizer.encoder[tokenizer.INIT + "Agent"]] = [
            tokenizer.encoder[tokenizer.INIT + entity] for entity in
            [entity_dict['PER'], entity_dict['ORG'], entity_dict['GPE']]]

        role_entity_type[tokenizer.encoder[tokenizer.INIT + "Victim"]] = [
            tokenizer.encoder[tokenizer.INIT + entity] for entity in
            [entity_dict['PER']]]

        role_entity_type[tokenizer.encoder[tokenizer.INIT + "Instrument"]] = [
            tokenizer.encoder[tokenizer.INIT + entity] for entity in
            [entity_dict['WEA'], entity_dict['VEH']]]

        role_entity_type[tokenizer.encoder[tokenizer.INIT + "Artifact"]] = [
            tokenizer.encoder[tokenizer.INIT + entity] for entity in
            [entity_dict['PER'], entity_dict['VEH'], entity_dict['WEA']]]

        role_entity_type[tokenizer.encoder[tokenizer.INIT + "Vehicle"]] = [
            tokenizer.encoder[tokenizer.INIT + entity] for entity in
            [entity_dict['VEH']]]

        role_entity_type[tokenizer.encoder[tokenizer.INIT + "Price"]] = [
            tokenizer.encoder[tokenizer.INIT + entity] for entity in
            [entity_dict['Numeric']]]

        role_entity_type[tokenizer.encoder[tokenizer.INIT + "Origin"]] = [
            tokenizer.encoder[tokenizer.INIT + entity] for entity in
            [entity_dict['GPE'], entity_dict['LOC'], entity_dict['FAC']]]

        role_entity_type[tokenizer.encoder[tokenizer.INIT + "Destination"]] = [
            tokenizer.encoder[tokenizer.INIT + entity] for entity in
            [entity_dict['GPE'], entity_dict['LOC'], entity_dict['FAC']]]

        role_entity_type[tokenizer.encoder[tokenizer.INIT + "Buyer"]] = [
            tokenizer.encoder[tokenizer.INIT + entity] for entity in
            [entity_dict['PER'], entity_dict['ORG'], entity_dict['GPE']]]

        role_entity_type[tokenizer.encoder[tokenizer.INIT + "Seller"]] = [
            tokenizer.encoder[tokenizer.INIT + entity] for entity in
            [entity_dict['PER'], entity_dict['ORG'], entity_dict['GPE']]]

        role_entity_type[tokenizer.encoder[tokenizer.INIT + "Beneficiary"]] = [
            tokenizer.encoder[tokenizer.INIT + entity] for entity in
            [entity_dict['PER'], entity_dict['ORG'], entity_dict['GPE']]]

        role_entity_type[tokenizer.encoder[tokenizer.INIT + "Giver"]] = [
            tokenizer.encoder[tokenizer.INIT + entity] for entity in
            [entity_dict['PER'], entity_dict['ORG'], entity_dict['GPE']]]

        role_entity_type[tokenizer.encoder[tokenizer.INIT + "Recipient"]] = [
            tokenizer.encoder[tokenizer.INIT + entity] for entity in
            [entity_dict['PER'], entity_dict['ORG'], entity_dict['GPE']]]

        role_entity_type[tokenizer.encoder[tokenizer.INIT + "Money"]] = [
            tokenizer.encoder[tokenizer.INIT + entity] for entity in
            [entity_dict['Numeric']]]

        role_entity_type[tokenizer.encoder[tokenizer.INIT + "Org"]] = [
            tokenizer.encoder[tokenizer.INIT + entity] for entity in
            [entity_dict['ORG']]]

        role_entity_type[tokenizer.encoder[tokenizer.INIT + "Attacker"]] = [
            tokenizer.encoder[tokenizer.INIT + entity] for entity in
            [entity_dict['ORG'], entity_dict['PER'], entity_dict['GPE']]]

        role_entity_type[tokenizer.encoder[tokenizer.INIT + "Target"]] = [
            tokenizer.encoder[tokenizer.INIT + entity] for entity in
            [entity_dict['ORG'], entity_dict['PER'], entity_dict['VEH'], entity_dict['FAC'], entity_dict['WEA']]]

        role_entity_type[tokenizer.encoder[tokenizer.INIT + "Entity"]] = [
            tokenizer.encoder[tokenizer.INIT + entity] for entity in
            [entity_dict['PER'], entity_dict['ORG'], entity_dict['GPE']]]

        role_entity_type[tokenizer.encoder[tokenizer.INIT + "Position"]] = [
            tokenizer.encoder[tokenizer.INIT + entity] for entity in
            [entity_dict['Job-Title']]]

        role_entity_type[tokenizer.encoder[tokenizer.INIT + "Crime"]] = [
            tokenizer.encoder[tokenizer.INIT + entity] for entity in
            [entity_dict['Crime']]]

        role_entity_type[tokenizer.encoder[tokenizer.INIT + "Defendant"]] = [
            tokenizer.encoder[tokenizer.INIT + entity] for entity in
            [entity_dict['PER'], entity_dict['ORG'], entity_dict['GPE']]]

        role_entity_type[tokenizer.encoder[tokenizer.INIT + "Prosecutor"]] = [
            tokenizer.encoder[tokenizer.INIT + entity] for entity in
            [entity_dict['PER'], entity_dict['ORG'], entity_dict['GPE']]]

        role_entity_type[tokenizer.encoder[tokenizer.INIT + "Adjudicator"]] = [
            tokenizer.encoder[tokenizer.INIT + entity] for entity in
            [entity_dict['PER'], entity_dict['ORG'], entity_dict['GPE']]]

        role_entity_type[tokenizer.encoder[tokenizer.INIT + "Plaintiff"]] = [
            tokenizer.encoder[tokenizer.INIT + entity] for entity in
            [entity_dict['PER'], entity_dict['ORG'], entity_dict['GPE']]]

        role_entity_type[tokenizer.encoder[tokenizer.INIT + "Sentence"]] = [
            tokenizer.encoder[tokenizer.INIT + entity] for entity in
            [entity_dict['Sentence']]]
        return event_type_list, event_ids, arg_ids, arg_indexes_ids, role_ids, label_dict, role_entity_type, y_args_ids, y_entity_ids, entity_arg_ids, entity_arg_dict, total_args_number, arg_candidates, entity_type_candidates, entity_type_arg_dict

    def get_seq(self, inputs, tokenizer):
        seq = []
        for token in inputs[0]:
            seq.append(tokenizer.decoder[int(token)])
        a = 0
        return seq

    def get_topk_graphs(self, total_graphs, beam_size):
        sorted_list = sorted(total_graphs, key=lambda x: x[2], reverse=True)
        return sorted_list[0:beam_size]

    def get_topk_graphs_for_stop(self, total_graphs, beam_size):
        sorted_list = sorted(total_graphs, key=lambda x: x[-1], reverse=True)
        return sorted_list[0:beam_size]

    def predict_beam_search(self, input_ids,
                            y,
                            cur_len,
                            max_length,
                            min_length,
                            do_sample,
                            temperature,
                            top_k,
                            top_p,
                            repetition_penalty,
                            no_repeat_ngram_size,
                            bad_words_ids,
                            bos_token_id,
                            pad_token_id,
                            eos_token_id,
                            stop_token_id,
                            decoder_start_token_id,
                            batch_size,
                            encoder_outputs,
                            attention_mask,
                            use_cache,
                            model_specific_kwargs, ):
        # length of generated sentences / unfinished sentences
        #
        unfinished_sents = input_ids.new(batch_size).fill_(1)
        sent_lengths = input_ids.new(batch_size).fill_(max_length)

        past = encoder_outputs  # defined for encoder-decoder models, None for decoder-only models
        # 获取限定的id
        tokenizer = model_specific_kwargs['tokenizer']
        entitys = model_specific_kwargs['entitys']
        event_type_list, event_ids, arg_ids, arg_indexes, role_ids, label_dict, role_entity_type, y_args_ids, y_entity_ids, entity_arg_ids, entity_arg_dict, total_args_number, arg_candidates, entity_type_candidates, entity_type_arg_dict = self.get_special_ids(
            tokenizer, y, entitys)
        argument_count = 0
        y_args_ids = arg_candidates
        y_entity_ids = entity_type_candidates
        entity_arg_dict = entity_type_arg_dict
        total_arg_num = len(y_args_ids)
        # total_arg_num = num
        e_list = []
        stop_count = 0
        arg_indexes_list = []
        arg_list = []
        beam_size = 1
        init_score = torch.tensor(0)
        new_partial_graph = [input_ids, past, init_score, argument_count, arg_list, 0, 0]
        final_graphs = []
        event_flag = 0
        while cur_len < 10:
            if cur_len == 1:
                model_inputs = self.prepare_inputs_for_generation(
                    input_ids, past=past, attention_mask=attention_mask, use_cache=use_cache, **model_specific_kwargs
                )

                # print(input_ids.size())
                outputs = self(**model_inputs)
                # print(outputs[0].size())
                next_token_logits = outputs[0][:, -1, :]
                next_token_logits = F.softmax(next_token_logits, dim=-1)
                # if model has past, then set the past variable to speed up decoding
                if self._use_cache(outputs, use_cache):
                    past = outputs[1]
                classification_token_ids = [tokenizer.yes_token_id, tokenizer.no_token_id]
                next_token_logits = torch.index_select(next_token_logits, 1,
                                                       torch.tensor(classification_token_ids).cuda())
                next_token1 = torch.argmax(next_token_logits, dim=-1)
                next_token_prob, next_token_index = torch.topk(next_token_logits, 1, dim=1)
                next_token = [classification_token_ids[int(id)] for id in next_token_index.squeeze(0)]
                next_token = torch.tensor(next_token).cuda()
                if int(next_token) == classification_token_ids[1]:
                    next_token_candidatas = [eos_token_id]
                    new_input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)
                    final_graph = torch.cat([new_input_ids, torch.tensor([eos_token_id]).cuda().unsqueeze(-1)], dim=-1)
                    break
                else:
                    score = init_score + next_token_prob[0][0]
                    # score = init_score
                    next_token_candidatas = event_ids
                    new_input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)
                    new_partial_graph = [new_input_ids, past, score, argument_count, arg_list, 0, 0]
                cur_len += 1
            elif cur_len == 2:
                total_graphs = []
                graph = new_partial_graph
                ids_input_ids = graph[0]
                ids_past = graph[1]
                ids_score = graph[2]
                ids_argument_count = graph[3]
                ids_arg_list = graph[4]
                ids_e_token = graph[5]
                ids_entity_token = graph[6]
                model_inputs = self.prepare_inputs_for_generation(
                    ids_input_ids, past=ids_past, attention_mask=attention_mask, use_cache=use_cache,
                    **model_specific_kwargs
                )

                # print(input_ids.size())
                outputs = self(**model_inputs)
                # print(outputs[0].size())
                next_token_logits = outputs[0][:, -1, :]
                next_token_logits = F.softmax(next_token_logits, dim=-1)
                # if model has past, then set the past variable to speed up decoding
                if self._use_cache(outputs, use_cache):
                    ids_past = outputs[1]

                # repetition penalty from CTRL paper (https://arxiv.org/abs/1909.05858)
                if repetition_penalty != 1.0:
                    self.enforce_repetition_penalty_(next_token_logits, batch_size, 1, input_ids, repetition_penalty)

                if no_repeat_ngram_size > 0:
                    # calculate a list of banned tokens to prevent repetitively generating the same ngrams
                    # from fairseq: https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345
                    banned_tokens = calc_banned_ngram_tokens(input_ids, batch_size, no_repeat_ngram_size, cur_len)
                    for batch_idx in range(batch_size):
                        next_token_logits[batch_idx, banned_tokens[batch_idx]] = -float("inf")

                if bad_words_ids is not None:
                    # calculate a list of banned tokens according to bad words
                    banned_tokens = calc_banned_bad_words_ids(input_ids, bad_words_ids)

                    for batch_idx in range(batch_size):
                        next_token_logits[batch_idx, banned_tokens[batch_idx]] = -float("inf")

                # set eos token prob to zero if min_length is not reached
                if eos_token_id is not None and cur_len < min_length:
                    next_token_logits[:, eos_token_id] = -float("inf")
                next_token_logits = torch.index_select(next_token_logits, 1, torch.tensor(event_ids).cuda())
                # next_token1 = torch.argmax(next_token_logits, dim=-1)
                next_token_probs, next_token_indexes = torch.topk(next_token_logits, beam_size, dim=1)
                next_tokens = [event_ids[int(id)] for id in next_token_indexes.squeeze(0)]
                for i, next_token in enumerate(next_tokens):
                    next_token = torch.tensor([next_token]).cuda()
                    next_score = next_token_probs[0][i]
                    tokens_to_add = next_token

                    # add token and increase length by one

                    new_input_ids = torch.cat([ids_input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
                    score = ids_score + next_score
                    total_graphs.append([new_input_ids, ids_past, score, ids_argument_count, ids_arg_list, ids_e_token,
                                         ids_entity_token, arg_indexes])
                cur_len += 1
            else:
                partial_graph = total_graphs
                total_graph = []
                stop_graphs = []
                for ids, graph in enumerate(partial_graph):
                    ids_input_ids = graph[0]
                    last_token_whether_event = ids_input_ids[0][-1]
                    if last_token_whether_event in event_ids:
                        stop_flag = 0
                        event_flag = 1
                        e_cur_len = 0
                        e_new_graphs = [graph]
                        while e_cur_len < 50:
                            e_total_graph = []
                            graphs = e_new_graphs
                            stop_g_count = 0
                            for e_ids, e_graph in enumerate(graphs):
                                e_ids_input_ids = e_graph[0]
                                e_last_token = e_ids_input_ids[0][-1]
                                if int(e_last_token) == stop_token_id:
                                    e_total_graph.append(e_graph)
                                    stop_g_count += 1
                                    if stop_g_count == beam_size:
                                        break
                                    continue
                                e_ids_past = e_graph[1]
                                e_ids_score = e_graph[2]
                                e_ids_argument_count = e_graph[3]
                                e_ids_arg_list = e_graph[4]
                                e_ids_e_token = e_graph[5]
                                e_ids_entity_token = e_graph[6]
                                e_ids_arg_indexes = e_graph[7]
                                # last_token_whether_event = ids_input_ids[-1]
                                model_inputs = self.prepare_inputs_for_generation(
                                    e_ids_input_ids, past=e_ids_past, attention_mask=attention_mask,
                                    use_cache=use_cache,
                                    **model_specific_kwargs
                                )

                                # print(input_ids.size())
                                outputs = self(**model_inputs)
                                # print(outputs[0].size())
                                next_token_logits = outputs[0][:, -1, :]
                                next_token_logits = F.softmax(next_token_logits, dim=-1)
                                # if model has past, then set the past variable to speed up decoding
                                if self._use_cache(outputs, use_cache):
                                    e_ids_past = outputs[1]

                                # repetition penalty from CTRL paper (https://arxiv.org/abs/1909.05858)
                                if repetition_penalty != 1.0:
                                    self.enforce_repetition_penalty_(next_token_logits, batch_size, 1, input_ids,
                                                                     repetition_penalty)

                                if no_repeat_ngram_size > 0:
                                    # calculate a list of banned tokens to prevent repetitively generating the same ngrams
                                    # from fairseq: https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345
                                    banned_tokens = calc_banned_ngram_tokens(input_ids, batch_size,
                                                                             no_repeat_ngram_size,
                                                                             cur_len)
                                    for batch_idx in range(batch_size):
                                        next_token_logits[batch_idx, banned_tokens[batch_idx]] = -float("inf")

                                if bad_words_ids is not None:
                                    # calculate a list of banned tokens according to bad words
                                    banned_tokens = calc_banned_bad_words_ids(input_ids, bad_words_ids)

                                    for batch_idx in range(batch_size):
                                        next_token_logits[batch_idx, banned_tokens[batch_idx]] = -float("inf")

                                # set eos token prob to zero if min_length is not reached
                                if eos_token_id is not None and cur_len < min_length:
                                    next_token_logits[:, eos_token_id] = -float("inf")
                                flag = 0
                                e_last_token = e_ids_input_ids[0][-1]
                                e_last_last_token = e_ids_input_ids[0][-2]
                                if e_last_token in event_ids:
                                    if total_arg_num != 0:
                                        if e_ids_argument_count >= total_arg_num:  # 论元个数够了
                                            next_token_candidatas = [stop_token_id]
                                        else:
                                            e_ids_e_token = e_last_token
                                            next_token_candidatas = y_entity_ids + [stop_token_id]
                                    else:  # 无论元
                                        next_token_candidatas = [stop_token_id]
                                if e_last_token in role_ids and e_last_last_token in arg_indexes + arg_ids + entity_arg_ids:
                                    if e_ids_argument_count >= total_arg_num:
                                        next_token_candidatas = [stop_token_id]
                                    else:
                                        next_token_candidatas = y_entity_ids + [stop_token_id]
                                if e_last_token in arg_indexes:
                                    if int(e_last_token) in e_ids_input_ids[0][:-1]:
                                        next_token_candidatas = label_dict[int(e_ids_e_token)]
                                    else:  # 新的论元出现
                                        if len(y_args_ids) != 0:
                                            if entity_arg_dict[int(e_ids_entity_token)]:
                                                new_arg_ids = []
                                                for a in entity_arg_dict[int(e_ids_entity_token)]:
                                                    if a not in e_ids_arg_list:
                                                        new_arg_ids.append(a)
                                                if len(new_arg_ids) == 0:
                                                    new_arg_ids = entity_arg_dict[int(e_ids_entity_token)]
                                                # next_token_candidatas = entity_arg_dict[int(entity_token)]
                                                next_token_candidatas = new_arg_ids
                                            else:
                                                next_token_candidatas = y_args_ids

                                if e_last_token in arg_ids and e_last_last_token in arg_indexes:
                                    next_token_candidatas = label_dict[int(e_ids_e_token)]
                                if e_last_token in y_entity_ids and e_last_last_token in event_ids + role_ids:
                                    e_ids_entity_token = e_last_token
                                    flag = 1
                                    if e_ids_argument_count >= total_arg_num:

                                        next_token_candidatas = e_ids_arg_indexes
                                    else:
                                        next_token_candidatas = e_ids_arg_indexes
                                if e_last_token in [stop_token_id]:
                                    next_token_candidatas = event_ids + [eos_token_id]
                                next_token_logits = torch.index_select(next_token_logits, 1,
                                                                       torch.tensor(next_token_candidatas).cuda())
                                if len(next_token_candidatas) < beam_size:
                                    new_beam_size = len(next_token_candidatas)
                                else:
                                    new_beam_size = beam_size
                                if flag:
                                    new_beam_size = 1
                                next_token_probs, next_token_indexes = torch.topk(next_token_logits,
                                                                                  new_beam_size, dim=1)
                                next_tokens = [next_token_candidatas[int(id)] for id in
                                               next_token_indexes.squeeze(0)]
                                for i, next_token in enumerate(next_tokens):
                                    next_token = torch.tensor([next_token]).cuda()
                                    next_score = next_token_probs[0][i]
                                    tokens_to_add = next_token

                                    # add token and increase length by one

                                    new_input_ids = torch.cat([e_ids_input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
                                    score = e_ids_score + next_score
                                    if int(next_token) in y_args_ids and int(e_last_token) in arg_indexes:
                                        new_ids_argument_count = e_ids_argument_count + 1
                                        if int(next_token) not in e_ids_arg_list:
                                            new_list = e_ids_arg_list + [int(next_token)]
                                    else:
                                        new_ids_argument_count = e_ids_argument_count
                                        new_list = e_ids_arg_list
                                    # if tokens_to_add == stop_token_id:
                                    #     stop_graphs.append([new_input_ids, e_ids_past, score, new_ids_argument_count, new_list, e_ids_e_token,
                                    #      e_ids_entity_token])
                                    #     break
                                    if int(next_token) == stop_token_id:
                                        new_arg_indexes = arg_indexes
                                    else:
                                        new_arg_indexes = copy.deepcopy(e_ids_arg_indexes)
                                        if int(next_token) in arg_indexes:
                                            new_arg_indexes.remove(int(next_token))
                                    e_total_graph.append(
                                        [new_input_ids, e_ids_past, score, new_ids_argument_count, new_list,
                                         e_ids_e_token,
                                         e_ids_entity_token, new_arg_indexes])
                            #
                            for it in e_total_graph:
                                it_len = len(it[0][0]) - 1
                                it_score = it[2] / it_len
                                it.append(it_score)
                            e_new_graphs = self.get_topk_graphs_for_stop(e_total_graph, beam_size)
                            e_cur_len = e_cur_len + 1
                        for e_it in e_total_graph:
                            total_graph.append(e_it)
                    else:
                        stop_flag = 1
                        for ids, graph in enumerate([graph]):
                            ids_input_ids = graph[0]
                            last_token = ids_input_ids[0][-1]
                            if int(last_token) == eos_token_id:
                                total_graph.append(graph)
                                continue
                            ids_past = graph[1]
                            ids_score = graph[2]
                            ids_argument_count = graph[3]
                            ids_arg_list = graph[4]
                            ids_e_token = graph[5]
                            ids_entity_token = graph[6]
                            ids_arg_indexes = graph[7]
                            model_inputs = self.prepare_inputs_for_generation(
                                ids_input_ids, past=ids_past, attention_mask=attention_mask, use_cache=use_cache,
                                **model_specific_kwargs
                            )

                            # print(input_ids.size())
                            outputs = self(**model_inputs)
                            # print(outputs[0].size())
                            next_token_logits = outputs[0][:, -1, :]
                            next_token_logits = F.softmax(next_token_logits, dim=-1)
                            # if model has past, then set the past variable to speed up decoding
                            if self._use_cache(outputs, use_cache):
                                ids_past = outputs[1]

                            # repetition penalty from CTRL paper (https://arxiv.org/abs/1909.05858)
                            if repetition_penalty != 1.0:
                                self.enforce_repetition_penalty_(next_token_logits, batch_size, 1, input_ids,
                                                                 repetition_penalty)

                            if no_repeat_ngram_size > 0:
                                # calculate a list of banned tokens to prevent repetitively generating the same ngrams
                                # from fairseq: https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345
                                banned_tokens = calc_banned_ngram_tokens(input_ids, batch_size, no_repeat_ngram_size,
                                                                         cur_len)
                                for batch_idx in range(batch_size):
                                    next_token_logits[batch_idx, banned_tokens[batch_idx]] = -float("inf")

                            if bad_words_ids is not None:
                                # calculate a list of banned tokens according to bad words
                                banned_tokens = calc_banned_bad_words_ids(input_ids, bad_words_ids)

                                for batch_idx in range(batch_size):
                                    next_token_logits[batch_idx, banned_tokens[batch_idx]] = -float("inf")

                            # set eos token prob to zero if min_length is not reached
                            if eos_token_id is not None and cur_len < min_length:
                                next_token_logits[:, eos_token_id] = -float("inf")

                            flag = 0
                            if ids_argument_count == 4:
                                a = 0
                            last_token = ids_input_ids[0][-1]
                            last_last_token = ids_input_ids[0][-2]

                            if last_token in [stop_token_id]:
                                next_token_candidatas = event_ids + [eos_token_id]

                            next_token_logits = torch.index_select(next_token_logits, 1,
                                                                   torch.tensor(next_token_candidatas).cuda())
                            if len(next_token_candidatas) < beam_size:
                                new_beam_size = len(next_token_candidatas)
                            else:
                                new_beam_size = beam_size
                            # if flag:
                            #     new_beam_size=1
                            next_token_probs, next_token_indexes = torch.topk(next_token_logits, new_beam_size, dim=1)
                            next_tokens = [next_token_candidatas[int(id)] for id in next_token_indexes.squeeze(0)]

                            for i, next_token in enumerate(next_tokens):
                                next_token = torch.tensor([next_token]).cuda()
                                next_score = next_token_probs[0][i]
                                tokens_to_add = next_token

                                # add token and increase length by one

                                new_input_ids = torch.cat([ids_input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
                                score = ids_score + next_score
                                total_graph.append(
                                    [new_input_ids, ids_past, score, ids_argument_count, ids_arg_list, ids_e_token,
                                     ids_entity_token, ids_arg_indexes])
                        # total_graphs=self.get_topk_graphs(total_graph,beam_size)
                for it in total_graph:
                    it_len = len(it[0][0]) - 1
                    it_score = it[2] / it_len
                    it.append(it_score)
                total_graphs = self.get_topk_graphs_for_stop(total_graph, beam_size)
                cur_len = cur_len + 1

        if cur_len == 10:
            for g in total_graphs:
                l = (len(g[0][0]) - 1)
                #     if eos_token_id in g[0][0]:
                #         l=l-1
                g[2] = (g[2]) / l
            # if len(final_graphs) == 0:
            #     final_graphs = partial_graph
            final_graph = self.get_topk_graphs(total_graphs, 1)
            final_graph = final_graph[0][0]

        seqs = self.get_seq(final_graph, tokenizer)
        # if there are different sentences lengths in the batch, some batches have to be padded
        #
        return seqs, event_ids, arg_ids, arg_indexes, role_ids

    def get_topk_graphs_step(self, total_graphs, beam_size):
        sorted_list = sorted(total_graphs, key=lambda x: x[3], reverse=True)
        return sorted_list[0:beam_size]

    def predict_no_beam_search(self, input_ids,
                               y,
                               cur_len,
                               max_length,
                               min_length,
                               do_sample,
                               temperature,
                               top_k,
                               top_p,
                               repetition_penalty,
                               no_repeat_ngram_size,
                               bad_words_ids,
                               bos_token_id,
                               pad_token_id,
                               eos_token_id,
                               stop_token_id,
                               decoder_start_token_id,
                               batch_size,
                               encoder_outputs,
                               attention_mask,
                               use_cache,
                               model_specific_kwargs, ):
        # length of generated sentences / unfinished sentences
        unfinished_sents = input_ids.new(batch_size).fill_(1)
        sent_lengths = input_ids.new(batch_size).fill_(max_length)

        past = encoder_outputs  # defined for encoder-decoder models, None for decoder-only models
        # 获取限定的id
        tokenizer = model_specific_kwargs['tokenizer']
        entitys = model_specific_kwargs['entitys']
        # e_d_list = model_specific_kwargs['e_d']
        # e_d_id_list = self.get_event_ids(tokenizer, e_d_list)
        event_type_list, event_ids, arg_ids, arg_indexes, role_ids, label_dict, role_entity_type, y_args_ids, y_entity_ids, entity_arg_ids, entity_arg_dict, total_args_number, arg_candidates, entity_type_candidates, entity_type_arg_dict = self.get_special_ids(
            tokenizer, y, entitys)
        e_d_id_list = event_type_list
        argument_count = 0
        # total_arg_num = total_args_number
        # total_arg_num = len(y_args_ids)
        total_arg_num = len(arg_candidates)
        y_args_ids = arg_candidates
        # if not set(y_entity_ids).issubset(set(list(entity_arg_dict.keys()))):
        y_entity_ids = entity_type_candidates
        entity_arg_dict = entity_type_arg_dict
        e_list = []
        stop_count = 0
        arg_indexes_list = []
        arg_list = []
        new_arg_indexes = copy.deepcopy(arg_indexes)
        while cur_len < max_length:

            model_inputs = self.prepare_inputs_for_generation(
                input_ids, past=past, attention_mask=attention_mask, use_cache=use_cache, **model_specific_kwargs
            )

            # print(input_ids.size())
            outputs = self(**model_inputs)
            # print(outputs[0].size())
            next_token_logits = outputs[0][:, -1, :]

            # if model has past, then set the past variable to speed up decoding
            if self._use_cache(outputs, use_cache):
                past = outputs[1]

            # repetition penalty from CTRL paper (https://arxiv.org/abs/1909.05858)
            if repetition_penalty != 1.0:
                self.enforce_repetition_penalty_(next_token_logits, batch_size, 1, input_ids, repetition_penalty)

            if no_repeat_ngram_size > 0:
                # calculate a list of banned tokens to prevent repetitively generating the same ngrams
                # from fairseq: https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345
                banned_tokens = calc_banned_ngram_tokens(input_ids, batch_size, no_repeat_ngram_size, cur_len)
                for batch_idx in range(batch_size):
                    next_token_logits[batch_idx, banned_tokens[batch_idx]] = -float("inf")

            if bad_words_ids is not None:
                # calculate a list of banned tokens according to bad words
                banned_tokens = calc_banned_bad_words_ids(input_ids, bad_words_ids)

                for batch_idx in range(batch_size):
                    next_token_logits[batch_idx, banned_tokens[batch_idx]] = -float("inf")

            # set eos token prob to zero if min_length is not reached
            if eos_token_id is not None and cur_len < min_length:
                next_token_logits[:, eos_token_id] = -float("inf")

            if do_sample:
                # Temperature (higher temperature => more likely to sample low probability tokens)
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature
                # Top-p/top-k filtering
                next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
                # Sample
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                # ee 限定解码
                if tokenizer.dfs_linearization:
                    if cur_len == 1:
                        classification_token_ids = [tokenizer.yes_token_id, tokenizer.no_token_id]
                        if len(e_d_id_list) == 0:
                            next_token = [tokenizer.no_token_id]
                            next_token = torch.tensor(next_token).cuda()
                        else:
                            next_token = [tokenizer.yes_token_id]
                            next_token = torch.tensor(next_token).cuda()
                        if int(next_token) == classification_token_ids[1]:
                            next_token_candidatas = [eos_token_id]
                        else:
                            next_token_candidatas = e_d_id_list
                    else:
                        if cur_len == 2:
                            if next_token_candidatas == [eos_token_id]:
                                next_token = torch.tensor([eos_token_id]).cuda()
                            else:
                                next_token = e_d_id_list[0]
                                next_token = torch.tensor(next_token).cuda()
                            if int(next_token) in event_ids:
                                e_d_id_list.remove(int(next_token))
                        else:
                            if argument_count > total_arg_num:
                                break
                            last_token = input_ids[0][-1]
                            last_last_token = input_ids[0][-2]
                            if last_token in event_ids:
                                if total_arg_num != 0:
                                    if argument_count >= total_arg_num:  # 论元个数够了
                                        next_token_candidatas = [stop_token_id]
                                    else:
                                        e_token = last_token
                                        next_token_candidatas = y_entity_ids + [stop_token_id]
                                else:  # 无论元
                                    next_token_candidatas = [stop_token_id]
                            if last_token in role_ids and last_last_token in arg_indexes + arg_ids + entity_arg_ids:
                                if argument_count >= total_arg_num:
                                    next_token_candidatas = [stop_token_id]
                                else:
                                    next_token_candidatas = y_entity_ids + [stop_token_id]
                            if last_token in arg_indexes:
                                if int(last_token) in input_ids[0][:-1]:
                                    next_token_candidatas = label_dict[int(e_token)]
                                else:  # 新的论元出现
                                    new_arg_ids = []
                                    for a in entity_arg_dict[int(entity_token)]:
                                        if a not in arg_list:
                                            new_arg_ids.append(a)
                                    if len(y_args_ids) != 0:
                                        if entity_arg_dict[int(entity_token)]:
                                            if new_arg_ids:
                                                next_token_candidatas = new_arg_ids
                                            else:
                                                next_token_candidatas = entity_arg_dict[int(entity_token)]
                                        else:
                                            next_token_candidatas = y_args_ids
                                    argument_count += 1
                            if last_token in arg_ids + entity_arg_ids and last_last_token in arg_indexes:
                                next_token_candidatas = label_dict[int(e_token)]
                            if last_token in y_entity_ids and last_last_token in event_ids + role_ids:
                                entity_token = last_token
                                next_token_candidatas = new_arg_indexes
                            if last_token in [stop_token_id]:
                                if len(e_d_id_list) != 0:
                                    next_token_candidatas = [e_d_id_list[0]]
                                else:
                                    next_token_candidatas = [eos_token_id]

                            # role_ids+[eos_token_id]+event_ids
                            # print(next_token_logits.size())
                            next_token_logits = torch.index_select(next_token_logits, 1,
                                                                   torch.tensor(next_token_candidatas).cuda())
                            next_token_prob, next_token_index = torch.topk(next_token_logits, 1, dim=1)
                            next_token = [next_token_candidatas[int(id)] for id in next_token_index.squeeze(0)]
                            next_token = torch.tensor(next_token).cuda()
                            if int(next_token) in arg_indexes:
                                if int(next_token) not in arg_indexes_list:
                                    arg_indexes_list.append(int(next_token))
                            if int(next_token) in arg_ids + entity_arg_ids and int(last_token) in arg_indexes:
                                if int(next_token) not in arg_list:
                                    arg_list.append(int(next_token))
                            if int(next_token) == stop_token_id:
                                new_arg_indexes = copy.deepcopy(arg_indexes)
                            else:
                                if int(next_token) in arg_indexes:
                                    new_arg_indexes.remove(int(next_token))
                            if int(next_token) in event_ids:
                                e_d_id_list.remove(int(next_token))

                        # k_score, k_concept = torch.topk(all_score, beam_size, dim=1)
                        # k_concept = [event_types_indices[int(id)] for id in k_concept.squeeze(0)]
                ############################################################
                if tokenizer.bfs_linearization:
                    if cur_len == 1:
                        next_token_logits = torch.index_select(next_token_logits, 1, torch.tensor(event_ids).cuda())
                        next_token1 = torch.argmax(next_token_logits, dim=-1)
                        next_token_prob, next_token_index = torch.topk(next_token_logits, 1, dim=1)
                        next_token = [event_ids[int(id)] for id in next_token_index.squeeze(0)]
                        next_token = torch.tensor(next_token).cuda()
                        e_list.append(next_token[0])
                    else:
                        if argument_count > total_arg_num:
                            break
                        last_token = input_ids[0][-1]
                        if last_token in event_ids:
                            if total_arg_num != 0:
                                e_token = e_list[stop_count]
                                next_token_candidatas = event_ids + arg_indexes + [stop_token_id]
                            else:
                                next_token_candidatas = event_ids + [stop_token_id]
                        if last_token in role_ids:  # 事件类型结束
                            if argument_count > total_arg_num:
                                next_token_candidatas = [stop_token_id]
                            else:
                                next_token_candidatas = [arg_indexes] + [stop_token_id]
                        if last_token in arg_indexes:
                            if int(last_token) in input_ids[0][:-1]:
                                e_token = e_list[stop_count]
                                next_token_candidatas = label_dict[int(e_token)]
                            else:  # 新的论元出现
                                if len(y_args_ids) != 0:
                                    next_token_candidatas = y_args_ids
                                argument_count += 1
                        if last_token in arg_ids:
                            e_token = e_list[stop_count]
                            next_token_candidatas = label_dict[int(e_token)]
                        if last_token in [stop_token_id]:
                            if stop_count >= len(e_list):
                                next_token_candidatas = [eos_token_id]
                            else:
                                if argument_count >= total_arg_num:
                                    next_token_candidatas = [stop_token_id]
                                else:
                                    next_token_candidatas = [stop_token_id] + arg_indexes
                        next_token_logits = torch.index_select(next_token_logits, 1,
                                                               torch.tensor(next_token_candidatas).cuda())
                        next_token_prob, next_token_index = torch.topk(next_token_logits, 1, dim=1)
                        next_token = [next_token_candidatas[int(id)] for id in next_token_index.squeeze(0)]
                        next_token = torch.tensor(next_token).cuda()
                        if int(next_token) in event_ids:
                            e_list.append(next_token[0])
                        if int(next_token) in [stop_token_id]:
                            stop_count += 1

            # update generations and finished sentences
            if eos_token_id is not None:
                # pad finished sentences if eos_token_id exist
                tokens_to_add = next_token * unfinished_sents + (pad_token_id) * (1 - unfinished_sents)
            else:
                tokens_to_add = next_token

            # add token and increase length by one
            input_ids = torch.cat([input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
            cur_len = cur_len + 1

            if eos_token_id is not None:
                eos_in_sents = tokens_to_add == eos_token_id
                # if sentence is unfinished and the token to add is eos, sent_lengths is filled with current length
                is_sents_unfinished_and_token_to_add_is_eos = unfinished_sents.mul(eos_in_sents.long()).bool()
                sent_lengths.masked_fill_(is_sents_unfinished_and_token_to_add_is_eos, cur_len)
                # unfinished_sents is set to zero if eos in sentence
                unfinished_sents.mul_((~eos_in_sents).long())

            # stop when there is a </s> in each sentence, or if we exceed the maximul length
            if unfinished_sents.max() == 0:
                break

            # extend attention_mask for new generated input if only decoder
            # if self.config.is_encoder_decoder is False:
            #     attention_mask = torch.cat(
            #         [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
            #     )

        seqs = self.get_seq(input_ids, tokenizer)
        print(seqs)
        # if there are different sentences lengths in the batch, some batches have to be padded
        #
        return seqs, event_ids, arg_ids, arg_indexes, role_ids

    def forward(
            self,
            input_ids,
            attention_mask=None,
            encoder_outputs=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            decoder_cached_states=None,
            lm_labels=None,
            use_cache=False,
            **unused
    ):
        r"""
        lm_labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for computing the masked language modeling loss.
            Indices should either be in ``[0, ..., config.vocab_size]`` or -100 (see ``input_ids`` docstring).
            Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens
            with labels
            in ``[0, ..., config.vocab_size]``.

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.RobertaConfig`) and inputs:
        masked_lm_loss (`optional`, returned when ``lm_labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Masked language modeling loss.
        prediction_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`)
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

            # Mask filling only works for bart-large
            from transformers import BartTokenizer, BartForConditionalGeneration
            tokenizer = BartTokenizer.from_pretrained('bart-large')
            TXT = "My friends are <mask> but they eat too many carbs."
            model = BartForConditionalGeneration.from_pretrained('bart-large')
            input_ids = tokenizer.batch_encode_plus([TXT], return_tensors='pt')['input_ids']
            logits = model(input_ids)[0]
            masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
            probs = logits[0, masked_index].softmax(dim=0)
            values, predictions = probs.topk(5)
            tokenizer.decode(predictions).split()
            # ['good', 'great', 'all', 'really', 'very']
        """
        # outputs = self.model(
        #     input_ids,
        #     attention_mask=attention_mask,
        #     decoder_input_ids=decoder_input_ids,
        #     encoder_outputs=encoder_outputs,
        #     decoder_attention_mask=decoder_attention_mask,
        #     decoder_cached_states=decoder_cached_states,
        #     use_cache=use_cache,
        # )
        # lm_logits = F.linear(outputs[0][0], self.model.shared.weight, bias=self.final_logits_bias)
        # po_logits = outputs[0][1]
        # po_padding = torch.full_like(po_logits[:, :, 0:1], float('-inf'))
        # po_padding = po_padding.repeat(1, 1, 1024 - po_logits.size(-1))
        # po_logits = torch.cat([po_logits, po_padding], -1)
        # uni_logits = torch.cat([lm_logits, po_logits], -1)
        #
        # outputs = (uni_logits,) + outputs[1:]  # Add cache, hidden states and attention if they are here

        outputs = self.compute_logits(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            decoder_cached_states=decoder_cached_states,
            use_cache=use_cache,
        )

        if lm_labels is not None:
            uni_logits = outputs[0]
            masked_lm_loss = F.nll_loss(
                uni_logits.log_softmax(-1).contiguous().view(-1, uni_logits.size(-1)),
                lm_labels.contiguous().view(-1),
                ignore_index=self.pad_index)
            outputs = (masked_lm_loss,) + outputs

        return outputs

    def compute_logits(
            self,
            input_ids,
            attention_mask=None,
            encoder_outputs=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            decoder_cached_states=None,
            use_cache=False,
    ):
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            decoder_cached_states=decoder_cached_states,
            use_cache=use_cache,
        )

        lm_logits = F.linear(outputs[0][0], self.model.shared.weight, bias=self.final_logits_bias)
        po_logits = outputs[0][1]
        po_padding = torch.full_like(po_logits[:, :, 0:1], float('-inf'))
        po_padding = po_padding.repeat(1, 1, 1024 - po_logits.size(-1))
        po_logits = torch.cat([po_logits, po_padding], -1)
        uni_logits = torch.cat([lm_logits, po_logits], -1)
        outputs = (uni_logits,) + outputs[1:]  # Add cache, hidden states and attention if they are here
        return outputs

    @torch.no_grad()
    def generate(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            y: Optional[torch.LongTensor] = None,
            # entitys:
            max_length: Optional[int] = None,
            # tokenizer: tokenizer,
            min_length: Optional[int] = None,
            do_sample: Optional[bool] = None,
            early_stopping: Optional[bool] = None,
            num_beams: Optional[int] = None,
            temperature: Optional[float] = None,
            top_k: Optional[int] = None,
            top_p: Optional[float] = None,
            repetition_penalty: Optional[float] = None,
            bad_words_ids: Optional[Iterable[int]] = None,
            bos_token_id: Optional[int] = None,
            pad_token_id: Optional[int] = None,
            eos_token_id: Optional[int] = None,
            length_penalty: Optional[float] = None,
            no_repeat_ngram_size: Optional[int] = None,
            num_return_sequences: Optional[int] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            decoder_start_token_id: Optional[int] = None,
            use_cache: Optional[bool] = None,
            **model_specific_kwargs
    ) -> torch.LongTensor:
        r""" Generates sequences for models with a LM head. The method currently supports greedy decoding, beam-search decoding, sampling with temperature, sampling with top-k or nucleus sampling.

        Adapted in part from `Facebook's XLM beam search code`_.

        .. _`Facebook's XLM beam search code`:
           https://github.com/facebookresearch/XLM/blob/9e6f6814d17be4fe5b15f2e6c43eb2b2d76daeb4/src/model/transformer.py#L529


        Parameters:

            input_ids: (`optional`) `torch.LongTensor` of shape `(batch_size, sequence_length)`
                The sequence used as a prompt for the generation. If `None` the method initializes
                it as an empty `torch.LongTensor` of shape `(1,)`.

            max_length: (`optional`) int
                The max length of the sequence to be generated.  Between `min_length` and infinity. Default to 20.

            min_length: (`optional`) int
                The min length of the sequence to be generated.  Between 0 and infinity. Default to 0.

            do_sample: (`optional`) bool
                If set to `False` greedy decoding is used. Otherwise sampling is used. Defaults to `False` as defined in `configuration_utils.PretrainedConfig`.

            early_stopping: (`optional`) bool
                if set to `True` beam search is stopped when at least `num_beams` sentences finished per batch. Defaults to `False` as defined in `configuration_utils.PretrainedConfig`.

            num_beams: (`optional`) int
                Number of beams for beam search. Must be between 1 and infinity. 1 means no beam search. Default to 1.

            temperature: (`optional`) float
                The value used to module the next token probabilities. Must be strictly positive. Default to 1.0.

            top_k: (`optional`) int
                The number of highest probability vocabulary tokens to keep for top-k-filtering. Between 1 and infinity. Default to 50.

            top_p: (`optional`) float
                The cumulative probability of parameter highest probability vocabulary tokens to keep for nucleus sampling. Must be between 0 and 1. Default to 1.

            repetition_penalty: (`optional`) float
                The parameter for repetition penalty. Between 1.0 and infinity. 1.0 means no penalty. Default to 1.0.

            pad_token_id: (`optional`) int
                Padding token. Default to specicic model pad_token_id or None if it does not exist.

            bos_token_id: (`optional`) int
                BOS token. Defaults to `bos_token_id` as defined in the models config.

            eos_token_id: (`optional`) int
                EOS token. Defaults to `eos_token_id` as defined in the models config.

            length_penalty: (`optional`) float
                Exponential penalty to the length. Default to 1.

            no_repeat_ngram_size: (`optional`) int
                If set to int > 0, all ngrams of size `no_repeat_ngram_size` can only occur once.
            bad_words_ids: (`optional`) list of lists of int
                `bad_words_ids` contains tokens that are not allowed to be generated. In order to get the tokens of the words that should not appear in the generated text, use `tokenizer.encode(bad_word, add_prefix_space=True)`.

            num_return_sequences: (`optional`) int
                The number of independently computed returned sequences for each element in the batch. Default to 1.

            attention_mask (`optional`) obj: `torch.LongTensor` of same shape as `input_ids`
                Mask to avoid performing attention on padding token indices.
                Mask values selected in ``[0, 1]``:
                ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
                Defaults to `None`.

                `What are attention masks? <../glossary.html#attention-mask>`__

            decoder_start_token_id=None: (`optional`) int
                If an encoder-decoder model starts decoding with a different token than BOS.
                Defaults to `None` and is changed to `BOS` later.

            use_cache: (`optional`) bool
                If `use_cache` is True, past key values are used to speed up decoding if applicable to model. Defaults to `True`.

            model_specific_kwargs: (`optional`) dict
                Additional model specific kwargs will be forwarded to the `forward` function of the model.

        Return:

            output: `torch.LongTensor` of shape `(batch_size * num_return_sequences, sequence_length)`
                sequence_length is either equal to max_length or shorter if all batches finished early due to the `eos_token_id`

        Examples::

            tokenizer = AutoTokenizer.from_pretrained('distilgpt2')   # Initialize tokenizer
            model = AutoModelWithLMHead.from_pretrained('distilgpt2')    # Download model and configuration from S3 and cache.
            outputs = model.generate(max_length=40)  # do greedy decoding
            print('Generated: {}'.format(tokenizer.decode(outputs[0], skip_special_tokens=True)))

            tokenizer = AutoTokenizer.from_pretrained('openai-gpt')   # Initialize tokenizer
            model = AutoModelWithLMHead.from_pretrained('openai-gpt')    # Download model and configuration from S3 and cache.
            input_context = 'The dog'
            input_ids = tokenizer.encode(input_context, return_tensors='pt')  # encode input context
            outputs = model.generate(input_ids=input_ids, num_beams=5, num_return_sequences=3, temperature=1.5)  # generate 3 independent sequences using beam search decoding (5 beams) with sampling from initial context 'The dog'
            for i in range(3): #  3 output sequences were generated
                print('Generated {}: {}'.format(i, tokenizer.decode(outputs[i], skip_special_tokens=True)))

            tokenizer = AutoTokenizer.from_pretrained('distilgpt2')   # Initialize tokenizer
            model = AutoModelWithLMHead.from_pretrained('distilgpt2')    # Download model and configuration from S3 and cache.
            input_context = 'The dog'
            input_ids = tokenizer.encode(input_context, return_tensors='pt')  # encode input context
            outputs = model.generate(input_ids=input_ids, max_length=40, temperature=0.7, num_return_sequences=3)  # 3 generate sequences using by sampling
            for i in range(3): #  3 output sequences were generated
                print('Generated {}: {}'.format(i, tokenizer.decode(outputs[i], skip_special_tokens=True)))

            tokenizer = AutoTokenizer.from_pretrained('ctrl')   # Initialize tokenizer
            model = AutoModelWithLMHead.from_pretrained('ctrl')    # Download model and configuration from S3 and cache.
            input_context = 'Legal My neighbor is'  # "Legal" is one of the control codes for ctrl
            input_ids = tokenizer.encode(input_context, return_tensors='pt')  # encode input context
            outputs = model.generate(input_ids=input_ids, max_length=50, temperature=0.7, repetition_penalty=1.2)  # generate sequences
            print('Generated: {}'.format(tokenizer.decode(outputs[0], skip_special_tokens=True)))

            tokenizer = AutoTokenizer.from_pretrained('gpt2')   # Initialize tokenizer
            model = AutoModelWithLMHead.from_pretrained('gpt2')    # Download model and configuration from S3 and cache.
            input_context = 'My cute dog'  # "Legal" is one of the control codes for ctrl
            bad_words_ids = [tokenizer.encode(bad_word, add_prefix_space=True) for bad_word in ['idiot', 'stupid', 'shut up']]
            input_ids = tokenizer.encode(input_context, return_tensors='pt')  # encode input context
            outputs = model.generate(input_ids=input_ids, max_length=100, do_sample=True, bad_words_ids=bad_words_ids)  # generate sequences without allowing bad_words to be generated
        """

        # We cannot generate if the model does not have a LM head
        if self.get_output_embeddings() is None:
            raise AttributeError(
                "You tried to generate sequences with a model that does not have a LM Head."
                "Please use another model class (e.g. `OpenAIGPTLMHeadModel`, `XLNetLMHeadModel`, `GPT2LMHeadModel`, `CTRLLMHeadModel`, `T5WithLMHeadModel`, `TransfoXLLMHeadModel`, `XLMWithLMHeadModel`, `BartForConditionalGeneration` )"
            )
        tokenizer = tokenizer = model_specific_kwargs['tokenizer']
        max_length = max_length if max_length is not None else self.config.max_length
        min_length = min_length if min_length is not None else self.config.min_length
        do_sample = do_sample if do_sample is not None else self.config.do_sample
        early_stopping = early_stopping if early_stopping is not None else self.config.early_stopping
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        num_beams = num_beams if num_beams is not None else self.config.num_beams
        temperature = temperature if temperature is not None else self.config.temperature
        top_k = top_k if top_k is not None else self.config.top_k
        top_p = top_p if top_p is not None else self.config.top_p
        repetition_penalty = repetition_penalty if repetition_penalty is not None else self.config.repetition_penalty
        bos_token_id = bos_token_id if bos_token_id is not None else self.config.bos_token_id
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        stop_token_id = self.config.stop_token_id
        length_penalty = length_penalty if length_penalty is not None else self.config.length_penalty
        no_repeat_ngram_size = (
            no_repeat_ngram_size if no_repeat_ngram_size is not None else self.config.no_repeat_ngram_size
        )
        bad_words_ids = bad_words_ids if bad_words_ids is not None else self.config.bad_words_ids
        num_return_sequences = (
            num_return_sequences if num_return_sequences is not None else self.config.num_return_sequences
        )
        decoder_start_token_id = (
            decoder_start_token_id if decoder_start_token_id is not None else self.config.decoder_start_token_id
        )

        if input_ids is not None:
            # print(input_ids.size())
            batch_size = input_ids.shape[0]  # overriden by the input batch_size
        else:
            batch_size = 1

        assert isinstance(max_length, int) and max_length > 0, "`max_length` should be a strictly positive integer."
        assert isinstance(min_length, int) and min_length >= 0, "`min_length` should be a positive integer."
        assert isinstance(do_sample, bool), "`do_sample` should be a boolean."
        assert isinstance(early_stopping, bool), "`early_stopping` should be a boolean."
        assert isinstance(use_cache, bool), "`use_cache` should be a boolean."
        assert isinstance(num_beams, int) and num_beams > 0, "`num_beams` should be a strictly positive integer."
        assert temperature > 0, "`temperature` should be strictly positive."
        assert isinstance(top_k, int) and top_k >= 0, "`top_k` should be a positive integer."
        assert 0 <= top_p <= 1, "`top_p` should be between 0 and 1."
        assert repetition_penalty >= 1.0, "`repetition_penalty` should be >= 1."
        assert input_ids is not None or (
                isinstance(bos_token_id, int) and bos_token_id >= 0
        ), "If input_ids is not defined, `bos_token_id` should be a positive integer."
        assert pad_token_id is None or (
                isinstance(pad_token_id, int) and (pad_token_id >= 0)
        ), "`pad_token_id` should be a positive integer."
        assert (eos_token_id is None) or (
                isinstance(eos_token_id, int) and (eos_token_id >= 0)
        ), "`eos_token_id` should be a positive integer."
        assert length_penalty > 0, "`length_penalty` should be strictly positive."
        assert (
                isinstance(no_repeat_ngram_size, int) and no_repeat_ngram_size >= 0
        ), "`no_repeat_ngram_size` should be a positive integer."
        assert (
                isinstance(num_return_sequences, int) and num_return_sequences > 0
        ), "`num_return_sequences` should be a strictly positive integer."
        assert (
                bad_words_ids is None or isinstance(bad_words_ids, list) and isinstance(bad_words_ids[0], list)
        ), "`bad_words_ids` is either `None` or a list of lists of tokens that should not be generated"

        if input_ids is None:
            assert isinstance(bos_token_id, int) and bos_token_id >= 0, (
                "you should either supply a context to complete as `input_ids` input "
                "or a `bos_token_id` (integer >= 0) as a first token to start the generation."
            )
            input_ids = torch.full(
                (batch_size, 1), bos_token_id, dtype=torch.long, device=next(self.parameters()).device,
            )
        else:
            assert input_ids.dim() == 2, "Input prompt should be of shape (batch_size, sequence length)."

        # not allow to duplicate outputs when greedy decoding
        if do_sample is False:
            if num_beams == 1:
                # no_beam_search greedy generation conditions
                assert (
                        num_return_sequences == 1
                ), "Greedy decoding will always produce the same output for num_beams == 1 and num_return_sequences > 1. Please set num_return_sequences = 1"

            else:
                # beam_search greedy generation conditions
                assert (
                        num_beams >= num_return_sequences
                ), "Greedy beam search decoding cannot return more sequences than it has beams. Please set num_beams >= num_return_sequences"

        # create attention mask if necessary
        # TODO (PVP): this should later be handled by the forward fn() in each model in the future see PR 3140
        if (attention_mask is None) and (pad_token_id is not None) and (pad_token_id in input_ids):
            attention_mask = input_ids.ne(pad_token_id).long()
        elif attention_mask is None:
            attention_mask = input_ids.new_ones(input_ids.shape)

        # set pad_token_id to eos_token_id if not set. Important that this is done after
        # attention_mask is created
        if pad_token_id is None and eos_token_id is not None:
            logger.warning(
                "Setting `pad_token_id` to {} (first `eos_token_id`) to generate sequence".format(eos_token_id)
            )
            pad_token_id = eos_token_id

        # current position and vocab size
        if hasattr(self.config, "vocab_size"):
            vocab_size = self.config.vocab_size
        elif (
                self.config.is_encoder_decoder
                and hasattr(self.config, "decoder")
                and hasattr(self.config.decoder, "vocab_size")
        ):
            vocab_size = self.config.decoder.vocab_size

        vocab_size += 1024

        # set effective batch size and effective batch multiplier according to do_sample
        if do_sample:
            effective_batch_size = batch_size * num_return_sequences
            effective_batch_mult = num_return_sequences
        else:
            effective_batch_size = batch_size
            effective_batch_mult = 1

        if self.config.is_encoder_decoder:
            if decoder_start_token_id is None:
                decoder_start_token_id = bos_token_id

            assert (
                    decoder_start_token_id is not None
            ), "decoder_start_token_id or bos_token_id has to be defined for encoder-decoder generation"
            assert hasattr(self, "get_encoder"), "{} should have a 'get_encoder' function defined".format(self)
            assert callable(self.get_encoder), "{} should be a method".format(self.get_encoder)

            # get encoder and store encoder outputs
            encoder = self.get_encoder()

            encoder_outputs: tuple = encoder(input_ids, attention_mask=attention_mask)

        # Expand input ids if num_beams > 1 or num_return_sequences > 1
        if num_return_sequences > 1 or num_beams > 1:
            input_ids_len = input_ids.shape[-1]
            input_ids = input_ids.unsqueeze(1).expand(batch_size, effective_batch_mult * num_beams, input_ids_len)
            attention_mask = attention_mask.unsqueeze(1).expand(
                batch_size, effective_batch_mult * num_beams, input_ids_len
            )

            input_ids = input_ids.contiguous().view(
                effective_batch_size * num_beams, input_ids_len
            )  # shape: (batch_size * num_return_sequences * num_beams, cur_len)
            attention_mask = attention_mask.contiguous().view(
                effective_batch_size * num_beams, input_ids_len
            )  # shape: (batch_size * num_return_sequences * num_beams, cur_len)

        if self.config.is_encoder_decoder:
            # create empty decoder_input_ids
            # print(0)
            input_ids = torch.full(
                (effective_batch_size * num_beams, 1),
                decoder_start_token_id,
                dtype=torch.long,
                device=next(self.parameters()).device,
            )
            cur_len = 1

            assert (
                    batch_size == encoder_outputs[0].shape[0]
            ), f"expected encoder_outputs[0] to have 1st dimension bs={batch_size}, got {encoder_outputs[0].shape[0]} "

            # expand batch_idx to assign correct encoder output for expanded input_ids (due to num_beams > 1 and num_return_sequences > 1)
            expanded_batch_idxs = (
                torch.arange(batch_size)
                    .view(-1, 1)
                    .repeat(1, num_beams * effective_batch_mult)
                    .view(-1)
                    .to(input_ids.device)
            )
            # expand encoder_outputs
            encoder_outputs = (encoder_outputs[0].index_select(0, expanded_batch_idxs), *encoder_outputs[1:])

        else:
            encoder_outputs = None
            cur_len = input_ids.shape[-1]

        if num_beams > 1:
            output = self._generate_beam_search(
                input_ids,
                cur_len=cur_len,
                max_length=max_length,
                min_length=min_length,
                do_sample=do_sample,
                early_stopping=early_stopping,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                bad_words_ids=bad_words_ids,
                bos_token_id=bos_token_id,
                pad_token_id=pad_token_id,
                decoder_start_token_id=decoder_start_token_id,
                eos_token_id=eos_token_id,
                batch_size=effective_batch_size,
                num_return_sequences=num_return_sequences,
                length_penalty=length_penalty,
                num_beams=num_beams,
                vocab_size=vocab_size,
                encoder_outputs=encoder_outputs,
                attention_mask=attention_mask,
                use_cache=use_cache,
                model_specific_kwargs=model_specific_kwargs,
            )
        else:
            # output = self._generate_no_beam_search(
            #     input_ids,
            #     cur_len=cur_len,
            #     max_length=max_length,
            #     min_length=min_length,
            #     do_sample=do_sample,
            #     temperature=temperature,
            #     top_k=top_k,
            #     top_p=top_p,
            #     repetition_penalty=repetition_penalty,
            #     no_repeat_ngram_size=no_repeat_ngram_size,
            #     bad_words_ids=bad_words_ids,
            #     bos_token_id=bos_token_id,
            #     pad_token_id=pad_token_id,
            #     decoder_start_token_id=decoder_start_token_id,
            #     eos_token_id=eos_token_id,
            #     batch_size=effective_batch_size,
            #     encoder_outputs=encoder_outputs,
            #     attention_mask=attention_mask,
            #     use_cache=use_cache,
            #     model_specific_kwargs=model_specific_kwargs,
            # )

            # print(cur_len)
            output, event_ids, arg_ids, arg_indexes, role_ids = self.predict_beam_search(
                input_ids,
                y,
                cur_len=cur_len,
                max_length=max_length,
                min_length=min_length,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                bad_words_ids=bad_words_ids,
                bos_token_id=bos_token_id,
                pad_token_id=pad_token_id,
                stop_token_id=stop_token_id,
                decoder_start_token_id=decoder_start_token_id,
                eos_token_id=eos_token_id,
                batch_size=effective_batch_size,
                encoder_outputs=encoder_outputs,
                attention_mask=attention_mask,
                use_cache=use_cache,
                model_specific_kwargs=model_specific_kwargs,
            )

        return output, event_ids, arg_ids, arg_indexes, role_ids

    def _generate_beam_search(
            self,
            input_ids,
            cur_len,
            max_length,
            min_length,
            do_sample,
            early_stopping,
            temperature,
            top_k,
            top_p,
            repetition_penalty,
            no_repeat_ngram_size,
            bad_words_ids,
            bos_token_id,
            pad_token_id,
            eos_token_id,
            decoder_start_token_id,
            batch_size,
            num_return_sequences,
            length_penalty,
            num_beams,
            vocab_size,
            encoder_outputs,
            attention_mask,
            use_cache,
            model_specific_kwargs,
    ):
        """ Generate sequences for each example with beam search.
        """

        # generated hypotheses
        generated_hyps = [
            BeamHypotheses(num_beams, max_length, length_penalty, early_stopping=early_stopping)
            for _ in range(batch_size)
        ]

        # scores for each sentence in the beam
        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)

        # for greedy decoding it is made sure that only tokens of the first beam are considered to avoid sampling the exact same tokens three times
        if do_sample is False:
            beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view(-1)  # shape (batch_size * num_beams,)

        # cache compute states
        past = encoder_outputs  # defined for encoder-decoder models, None for decoder-only models

        # done sentences
        done = [False for _ in range(batch_size)]

        while cur_len < max_length:
            model_inputs = self.prepare_inputs_for_generation(
                input_ids, past=past, attention_mask=attention_mask, use_cache=use_cache, **model_specific_kwargs
            )
            outputs = self(**model_inputs)  # (batch_size * num_beams, cur_len, vocab_size)
            next_token_logits = outputs[0][:, -1, :]  # (batch_size * num_beams, vocab_size)

            # if model has past, then set the past variable to speed up decoding
            if self._use_cache(outputs, use_cache):
                past = outputs[1]

            # repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858)
            if repetition_penalty != 1.0:
                self.enforce_repetition_penalty_(
                    next_token_logits, batch_size, num_beams, input_ids, repetition_penalty,
                )

            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            if self.config.is_encoder_decoder and do_sample is False:
                # TODO (PVP) still a bit hacky here - there might be a better solution
                next_token_logits = self.prepare_logits_for_generation(
                    next_token_logits, cur_len=cur_len, max_length=max_length
                )

            scores = F.log_softmax(next_token_logits, dim=-1)  # (batch_size * num_beams, vocab_size)

            # set eos token prob to zero if min_length is not reached
            if eos_token_id is not None and cur_len < min_length:
                scores[:, eos_token_id] = -float("inf")

            if no_repeat_ngram_size > 0:
                # calculate a list of banned tokens to prevent repetitively generating the same ngrams
                num_batch_hypotheses = batch_size * num_beams
                # from fairseq: https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345
                banned_batch_tokens = calc_banned_ngram_tokens(
                    input_ids, num_batch_hypotheses, no_repeat_ngram_size, cur_len
                )
                for i, banned_tokens in enumerate(banned_batch_tokens):
                    scores[i, banned_tokens] = -float("inf")

            if bad_words_ids is not None:
                # calculate a list of banned tokens according to bad words
                banned_tokens = calc_banned_bad_words_ids(input_ids, bad_words_ids)

                for i, banned_tokens in enumerate(banned_tokens):
                    scores[i, banned_tokens] = -float("inf")

            assert scores.shape == (batch_size * num_beams, vocab_size), "Shapes of scores: {} != {}".format(
                scores.shape, (batch_size * num_beams, vocab_size)
            )

            if do_sample:
                _scores = scores + beam_scores[:, None].expand_as(scores)  # (batch_size * num_beams, vocab_size)
                # Top-p/top-k filtering
                _scores = top_k_top_p_filtering(
                    _scores, top_k=top_k, top_p=top_p, min_tokens_to_keep=2
                )  # (batch_size * num_beams, vocab_size)
                # re-organize to group the beam together to sample from all beam_idxs
                _scores = _scores.contiguous().view(
                    batch_size, num_beams * vocab_size
                )  # (batch_size, num_beams * vocab_size)

                # Sample 2 next tokens for each beam (so we have some spare tokens and match output of greedy beam search)
                probs = F.softmax(_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=2 * num_beams)  # (batch_size, num_beams * 2)
                # Compute next scores
                next_scores = torch.gather(_scores, -1, next_tokens)  # (batch_size, num_beams * 2)
                # sort the sampled vector to make sure that the first num_beams samples are the best
                next_scores, next_scores_indices = torch.sort(next_scores, descending=True, dim=1)
                next_tokens = torch.gather(next_tokens, -1, next_scores_indices)  # (batch_size, num_beams * 2)

            else:
                next_scores = scores + beam_scores[:, None].expand_as(scores)  # (batch_size * num_beams, vocab_size)

                # re-organize to group the beam together (we are keeping top hypothesis accross beams)
                next_scores = next_scores.view(
                    batch_size, num_beams * vocab_size
                )  # (batch_size, num_beams * vocab_size)

                next_scores, next_tokens = torch.topk(next_scores, 2 * num_beams, dim=1, largest=True, sorted=True)

            assert next_scores.size() == next_tokens.size() == (batch_size, 2 * num_beams)

            # next batch beam content
            next_batch_beam = []

            # for each sentence
            for batch_idx in range(batch_size):

                # if we are done with this sentence
                if done[batch_idx]:
                    assert (
                            len(generated_hyps[batch_idx]) >= num_beams
                    ), "Batch can only be done if at least {} beams have been generated".format(num_beams)
                    assert (
                            eos_token_id is not None and pad_token_id is not None
                    ), "generated beams >= num_beams -> eos_token_id and pad_token have to be defined"
                    next_batch_beam.extend([(0, pad_token_id, 0)] * num_beams)  # pad the batch
                    continue

                # next sentence beam content
                next_sent_beam = []

                # next tokens for this sentence
                for beam_token_rank, (beam_token_id, beam_token_score) in enumerate(
                        zip(next_tokens[batch_idx], next_scores[batch_idx])
                ):
                    # get beam and token IDs
                    beam_id = beam_token_id // vocab_size
                    token_id = beam_token_id % vocab_size

                    effective_beam_id = batch_idx * num_beams + beam_id
                    # add to generated hypotheses if end of sentence or last iteration
                    if (eos_token_id is not None) and (token_id.item() == eos_token_id):
                        # if beam_token does not belong to top num_beams tokens, it should not be added
                        is_beam_token_worse_than_top_num_beams = beam_token_rank >= num_beams
                        if is_beam_token_worse_than_top_num_beams:
                            continue
                        generated_hyps[batch_idx].add(
                            input_ids[effective_beam_id].clone(), beam_token_score.item(),
                        )
                    else:
                        # add next predicted token if it is not eos_token
                        next_sent_beam.append((beam_token_score, token_id, effective_beam_id))

                    # the beam for next step is full
                    if len(next_sent_beam) == num_beams:
                        break

                # Check if were done so that we can save a pad step if all(done)
                done[batch_idx] = done[batch_idx] or generated_hyps[batch_idx].is_done(
                    next_scores[batch_idx].max().item(), cur_len=cur_len
                )

                # update next beam content
                assert len(next_sent_beam) == num_beams, "Beam should always be full"
                next_batch_beam.extend(next_sent_beam)
                assert len(next_batch_beam) == num_beams * (batch_idx + 1)

            # stop when we are done with each sentence
            if all(done):
                break

            # sanity check / prepare next batch
            assert len(next_batch_beam) == batch_size * num_beams
            beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
            beam_tokens = input_ids.new([x[1] for x in next_batch_beam])
            beam_idx = input_ids.new([x[2] for x in next_batch_beam])

            # re-order batch and update current length
            input_ids = input_ids[beam_idx, :]
            input_ids = torch.cat([input_ids, beam_tokens.unsqueeze(1)], dim=-1)
            cur_len = cur_len + 1

            # re-order internal states
            if past is not None:
                past = self._reorder_cache(past, beam_idx)

            # extend attention_mask for new generated input if only decoder
            if self.config.is_encoder_decoder is False:
                attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                )

        # finalize all open beam hypotheses and end to generated hypotheses
        for batch_idx in range(batch_size):
            if done[batch_idx]:
                continue

            # test that beam scores match previously calculated scores if not eos and batch_idx not done
            if eos_token_id is not None and all(
                    (token_id % vocab_size).item() is not eos_token_id for token_id in next_tokens[batch_idx]
            ):
                assert torch.all(
                    next_scores[batch_idx, :num_beams] == beam_scores.view(batch_size, num_beams)[batch_idx]
                ), "If batch_idx is not done, final next scores: {} have to equal to accumulated beam_scores: {}".format(
                    next_scores[:, :num_beams][batch_idx], beam_scores.view(batch_size, num_beams)[batch_idx],
                )

            # need to add best num_beams hypotheses to generated hyps
            for beam_id in range(num_beams):
                effective_beam_id = batch_idx * num_beams + beam_id
                final_score = beam_scores[effective_beam_id].item()
                final_tokens = input_ids[effective_beam_id]
                generated_hyps[batch_idx].add(final_tokens, final_score)

        # depending on whether greedy generation is wanted or not define different output_batch_size and output_num_return_sequences_per_batch
        output_batch_size = batch_size if do_sample else batch_size * num_return_sequences
        output_num_return_sequences_per_batch = 1 if do_sample else num_return_sequences

        # select the best hypotheses
        sent_lengths = input_ids.new(output_batch_size)
        best = []

        # retrieve best hypotheses
        for i, hypotheses in enumerate(generated_hyps):
            sorted_hyps = sorted(hypotheses.beams, key=lambda x: x[0])
            for j in range(output_num_return_sequences_per_batch):
                effective_batch_idx = output_num_return_sequences_per_batch * i + j
                best_hyp = sorted_hyps.pop()[1]
                sent_lengths[effective_batch_idx] = len(best_hyp)
                best.append(best_hyp)

        # shorter batches are filled with pad_token
        if sent_lengths.min().item() != sent_lengths.max().item():
            assert pad_token_id is not None, "`Pad_token_id` has to be defined"
            sent_max_len = min(sent_lengths.max().item() + 1, max_length)
            decoded = input_ids.new(output_batch_size, sent_max_len).fill_(pad_token_id)

            # fill with hypothesis and eos_token_id if necessary
            for i, hypo in enumerate(best):
                decoded[i, : sent_lengths[i]] = hypo
                if sent_lengths[i] < max_length:
                    decoded[i, sent_lengths[i]] = eos_token_id
        else:
            # none of the hypotheses have an eos_token
            assert (len(hypo) == max_length for hypo in best)
            decoded = torch.stack(best).type(torch.long).to(next(self.parameters()).device)

        return decoded

    @staticmethod
    def _reorder_cache(past: Tuple, beam_idx: Tensor) -> Tuple[Tensor]:
        return tuple(layer_past.index_select(1, beam_idx) for layer_past in past)

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        old_num_tokens = self.model.shared.num_embeddings
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self.model.shared = new_embeddings
        self._resize_final_logits_bias(new_num_tokens, old_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int, old_num_tokens: int) -> None:
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    def prepare_inputs_for_generation(self, decoder_input_ids, past, attention_mask, use_cache, **kwargs):
        assert past is not None, "past has to be defined for encoder_outputs"

        # first step, decoder_cached_states are empty
        if not past[1]:
            encoder_outputs, decoder_cached_states = past, None
        else:
            encoder_outputs, decoder_cached_states = past
        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "decoder_cached_states": decoder_cached_states,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    def prepare_logits_for_generation(self, logits, cur_len, max_length):
        # if cur_len == 1:
        #    self._force_token_ids_generation(logits, self.config.bos_token_id)
        if cur_len == max_length - 1 and self.config.eos_token_id is not None:
            self._force_token_ids_generation(logits, self.config.eos_token_id)
        return logits

    def _force_token_ids_generation(self, scores, token_ids) -> None:
        """force one of token_ids to be generated by setting prob of all other tokens to 0"""
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        all_but_token_ids_mask = torch.tensor(
            [x for x in range(self.config.vocab_size) if x not in token_ids],
            dtype=torch.long,
            device=next(self.parameters()).device,
        )
        assert len(scores.shape) == 2, "scores should be of rank 2 with shape: [batch_size, vocab_size]"
        scores[:, all_but_token_ids_mask] = -float("inf")

    @staticmethod
    def _reorder_cache(past, beam_idx):
        ((enc_out, enc_mask), decoder_cached_states) = past
        reordered_past = []
        for layer_past in decoder_cached_states:
            # get the correct batch idx from decoder layer's batch dim for cross and self-attn
            layer_past_new = {
                attn_key: bart._reorder_buffer(attn_cache, beam_idx) for attn_key, attn_cache in layer_past.items()
            }
            reordered_past.append(layer_past_new)

        new_enc_out = enc_out if enc_out is None else enc_out.index_select(0, beam_idx)
        new_enc_mask = enc_mask if enc_mask is None else enc_mask.index_select(0, beam_idx)

        past = ((new_enc_out, new_enc_mask), reordered_past)
        return past

    def get_encoder(self):
        return self.model.encoder

    def get_output_embeddings(self):
        return bart._make_linear_from_emb(self.model.shared)  # make it on the fly
