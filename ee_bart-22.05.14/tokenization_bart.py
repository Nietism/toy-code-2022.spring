from pathlib import Path

import regex as re
import torch
from transformers import BartTokenizer

from ee_bart.linearization import ACETokens, ACELinearizer

Event_types = {"Movement:Transport": 1, "Personnel:Elect": 1, "Personnel:Start-Position": 1, "Personnel:Nominate": 1,
               "Conflict:Attack": 1, "Personnel:End-Position": 1, "Life:Die": 1, "Contact:Meet": 1, "Life:Marry": 1,
               "Contact:Phone-Write": 1, "Transaction:Transfer-Money": 1, "Justice:Sue": 1, "Conflict:Demonstrate": 1,
               "Justice:Fine": 1, "Life:Injure": 1, "Business:End-Org": 1, "Justice:Trial-Hearing": 1,
               "Business:Start-Org": 1, "Justice:Arrest-Jail": 1, "Transaction:Transfer-Ownership": 1,
               "Justice:Execute": 1, "Justice:Sentence": 1, "Life:Be-Born": 1, "Justice:Charge-Indict": 1,
               "Business:Declare-Bankruptcy": 1, "Justice:Convict": 1, "Justice:Release-Parole": 1, "Justice:Pardon": 1,
               "Justice:Appeal": 1, "Business:Merge-Org": 1, "Justice:Extradite": 1, "Life:Divorce": 1,
               "Justice:Acquit": 1}
roles = {'Vehicle': 86, 'Artifact': 738, 'Destination': 571, 'Agent': 430, 'Person': 699, 'Position': 140,
         'Entity': 881, 'Attacker': 707, 'Place': 1124, 'Time-Within': 849, 'Victim': 673, 'Origin': 191,
         'Time-At-Beginning': 20, 'Target': 518, 'Giver': 136, 'Recipient': 151, 'Plaintiff': 84, 'Org': 124,
         'Time-Holds': 78, 'Prosecutor': 27, 'Money': 88, 'Defendant': 378, 'Buyer': 104, 'Instrument': 308,
         'Beneficiary': 32, 'Time-Ending': 24, 'Seller': 45, 'Time-At-End': 16, 'Time-Before': 30, 'Time-Starting': 61,
         'Time-After': 27, 'Adjudicator': 103, 'Sentence': 78, 'Crime': 260, 'Price': 12}

entity_dict = {'VEH': 'Vehicle', 'PER': 'Person', 'LOC': 'Location', 'Job-Title': 'Job-Title',
               'ORG': 'Organization',
               'GPE': 'Geopolitical-Entity', 'Time': 'Time', 'FAC': 'Facility', 'Numeric': 'Numeric',
               'WEA': 'Weapons', 'TIM': 'Time',
               'Sentence': 'Sentence', 'Crime': 'Crime', 'Contact-Info': 'Contact-Information'}

event_type_pair = {"Business:End-Org": "Business:End-Organization", "Business:Start-Org": "Business:Start-Organization",
                   'Business:Merge-Org': 'Business:Merge-Organization'}


class ACEBartTokenizer(BartTokenizer):
    INIT = 'Ġ'

    ADDITIONAL = [
        ACETokens.Yes_N,
        ACETokens.No_N, ACETokens.ENTITY_N]

    def __init__(self, *args, use_pointer_tokens=True, collapse_name_ops=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.patterns = re.compile(
            r""" ?<[a-z]+:?\d*>| ?:[^\s]+|'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        self.linearizer = ACELinearizer(use_pointer_tokens=use_pointer_tokens, collapse_name_ops=collapse_name_ops,
                                        dfs_linearization=self.dfs_linearization,
                                        bfs_linearization=self.bfs_linearization,
                                        use_entity_type=self.use_entity_type,
                                        use_classification=self.use_classification
                                        )
        self.use_pointer_tokens = use_pointer_tokens
        self.collapse_name_ops = collapse_name_ops
        self.recategorizations = set()
        self.modified = 0
        self.event_types = []
        self.event_type = []
        self.event_subtype = []
        self.arg_indexes = []
        self.roles = []
        self.args = []
        self.entity = []
        self.entity_arg = []

    @classmethod
    def from_pretrained(cls, pretrained_model_path, vocab_path, dfs_linearization,
                        bfs_linearization, use_entity_type, use_classification, *args, **kwargs):
        cls.dfs_linearization = dfs_linearization
        cls.bfs_linearization = bfs_linearization
        cls.use_entity_type = use_entity_type
        cls.use_classification = use_classification
        cls.vocab_path = vocab_path
        inst = super().from_pretrained(pretrained_model_path)
        inst.init_amr_vocabulary()
        return inst

    def init_amr_vocabulary(self):
        for tok in [self.bos_token, self.eos_token, self.pad_token, '<mask>', '<unk>', ]:
            ntok = self.INIT + tok
            i = self.encoder[tok]
            self.decoder[i] = ntok
            del self.encoder[tok]
            self.encoder[ntok] = i
        tokens = []
        for tok in Path(self.vocab_path + 'event_type.txt').read_text().strip().splitlines():
            # if tok in event_type_pair:
            #     tok = event_type_pair[tok]
            tokens.append(tok)
        for type in Event_types:
            self.event_types.append(type)
        for tok in Path(self.vocab_path + 'entity.txt').read_text().strip().splitlines():
            tokens.append(tok)
            self.entity_arg.append(tok)
        # for type in Event_types:
        #     type = type.split(":")
        #     if type[0] not in self.event_type:
        #         new_type = type[0]
        #         self.event_type.append(new_type)
        #         tokens.append(new_type)
        #     if type[1] not in self.event_subtype:
        #         new_subtype = type[1]
        #         self.event_subtype.append(new_subtype)
        #         tokens.append(new_subtype)
        if self.use_entity_type:
            for tok in Path(self.vocab_path + 'recategorization.txt').read_text().strip().splitlines():
                tokens.append(tok)
                self.entity.append(tok)
        for line in Path(self.vocab_path + 'argument.txt').read_text().strip().splitlines():
            tok, count = line.split()
            tokens.append(tok)
            if tok in roles:
                self.roles.append(tok)
            else:
                self.args.append(tok)
        # for tok in Path(self.vocab_path + 'arg.txt').read_text().strip().splitlines():
        #     self.args.append(tok)
        #     if tok not in tokens:
        #         tokens.append(tok)
        if self.use_pointer_tokens:
            for cnt in range(10):
                tokens.append(f"<Argument:{cnt}>")
                self.arg_indexes.append(f"<Argument:{cnt}>")
        if self.use_classification:
            tokens += self.ADDITIONAL
        tokens += [ACETokens.STOP_N]
        tokens = [self.INIT + t if t[0] not in ('_', '-') else t for t in tokens]
        tokens = [t for t in tokens if t not in self.encoder]
        self.old_enc_size = old_enc_size = len(self.encoder)
        for i, t in enumerate(tokens, start=old_enc_size):
            self.encoder[t] = i

        self.encoder = {k: i for i, (k, v) in enumerate(sorted(self.encoder.items(), key=lambda x: x[1]))}
        self.decoder = {v: k for k, v in sorted(self.encoder.items(), key=lambda x: x[1])}
        self.modified = len(tokens)

        self.bos_token = self.INIT + '<s>'
        self.pad_token = self.INIT + '<pad>'
        self.eos_token = self.INIT + '</s>'
        self.unk_token = self.INIT + '<unk>'
        self.stop_token_id = self.encoder[self.INIT + ACETokens.STOP_N]
        if self.use_classification:
            self.yes_token_id = self.encoder[self.INIT + ACETokens.Yes_N]
            self.no_token_id = self.encoder[self.INIT + ACETokens.No_N]

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        output = [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
        if token_ids_1 is None:
            return output
        return output + [self.eos_token_id] + token_ids_1 + [self.eos_token_id]

    def _tokenize(self, text):
        """ Tokenize a string. Modified in order to handle sentences with recategorization pointers"""
        bpe_tokens = []
        for tok_span in text.lstrip().split(' '):
            tok_span = tok_span.strip()
            recats = tok_span.rsplit('_', 1)
            if len(recats) == 2 and recats[0] in self.recategorizations and ('_' + recats[1]) in self.encoder:
                bpe_tokens.extend([self.INIT + recats[0], '_' + recats[1]])
            else:
                for token in re.findall(self.pat, ' ' + tok_span):
                    if token != ',':
                        token = "".join(
                            self.byte_encoder[b] for b in token.encode("utf-8")
                        )  # Maps all our bytes to unicode strings, avoiding controle tokens of the BPE (spaces in our case)
                        bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(" "))

        return bpe_tokens

    def _tok_bpe(self, token, add_space=True):
        # if add_space:
        #     token = ' ' + token.lstrip()
        tokk = []
        tok = token.strip()
        recats = tok.rsplit('_', 1)
        if len(recats) == 2 and recats[0] in self.recategorizations and ('_' + recats[1]) in self.encoder:
            tokk.extend([self.INIT + recats[0], '_' + recats[1]])
        else:
            for tok in self.patterns.findall(' ' + token):
                tok = "".join(
                    self.byte_encoder[b] for b in tok.encode("utf-8"))
                toks = self.bpe(tok).split(' ')
                tokk.extend(toks)
        return tokk

    def _get_nodes_and_backreferences(self, event_mentions):
        lin, triples = self.linearizer.linearize(event_mentions)
        if type(lin) == list:
            linearized_nodes, backreferences = lin, [0, 1, 2]
            triples = []
        else:
            linearized_nodes, backreferences = lin.nodes, lin.backreferences
        return linearized_nodes, backreferences, triples

    def tokenize_ee(self, event_mentions):
        linearized_nodes, backreferences, triples = self._get_nodes_and_backreferences(event_mentions)

        bpe_tokens = []
        bpe_backreferences = []
        counter = 0
        # 添加self.INIT
        for i, (backr, tokk) in enumerate(zip(backreferences, linearized_nodes)):
            is_in_enc = self.INIT + tokk in self.encoder
            if is_in_enc:
                bpe_toks = [self.INIT + tokk]
            else:
                bpe_toks = self._tok_bpe(tokk, add_space=True)

            bpe_tokens.append(bpe_toks)

            if i == backr:
                bpe_backr = list(range(counter, counter + len(bpe_toks)))
                counter += len(bpe_toks)
                bpe_backreferences.append(bpe_backr)
            else:
                bpe_backreferences.append(bpe_backreferences[backr][0:1])
                counter += 1
        bpe_tokens = [b for bb in bpe_tokens for b in bb]
        bpe_token_ids = [self.encoder.get(b, self.unk_token_id) for b in bpe_tokens]
        bpe_backreferences = [b for bb in bpe_backreferences for b in bb]
        return bpe_tokens, bpe_token_ids, bpe_backreferences, triples

    def batch_encode_sentences(self, sentences, device=torch.device('cpu')):
        sentences = [s for s in sentences]
        extra = {'sentences': sentences}
        batch = super().batch_encode_plus(sentences, return_tensors='pt', pad_to_max_length=True)
        batch = {k: v.to(device) for k, v in batch.items()}
        return batch, extra

    def linearize(self, event_mentions):
        shift = len(self.encoder)
        tokens, token_ids, backreferences, triples = self.tokenize_ee(event_mentions)
        extra = {'linearized_graphs': tokens, 'graphs': event_mentions, 'triples': triples}
        token_uni_ids = \
            [idx if i == b else b + shift for i, (idx, b) in enumerate(zip(token_ids, backreferences))]
        if token_uni_ids[-1] != (self.INIT + ACETokens.EOS_N):
            tokens.append(self.INIT + ACETokens.EOS_N)
            token_ids.append(self.eos_token_id)
            token_uni_ids.append(self.eos_token_id)
            backreferences.append(len(backreferences))
            return token_uni_ids, tokens, extra

    def batch_encode_graphs(self, graphs, device=torch.device('cpu')):
        linearized, extras = zip(*[self.linearize(g) for g in graphs])
        return self.batch_encode_graphs_from_linearized(linearized, extras, device=device)

    def batch_encode_graphs_from_linearized(self, linearized, extras=None, device=torch.device('cpu')):
        if extras is not None:
            batch_extra = {'linearized_graphs': [], 'graphs': []}
            for extra in extras:
                batch_extra['graphs'].append(extra['graphs'])
                batch_extra['linearized_graphs'].append(extra['linearized_graphs'])
        else:
            batch_extra = {}
        maxlen = 0
        batch = []
        for token_uni_ids in linearized:
            maxlen = max(len(token_uni_ids), maxlen)
            batch.append(token_uni_ids)
        batch = [x + [self.pad_token_id] * (maxlen - len(x)) for x in batch]
        batch = torch.tensor(batch).to(device)
        batch = {'decoder_input_ids': batch[:, :-1], 'lm_labels': batch[:, 1:]}
        return batch, batch_extra
