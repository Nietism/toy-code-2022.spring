import json
import logging
import random

import torch
from cached_property import cached_property
from torch.utils.data import Dataset

Entity_Dict = {'VEH': 'Vehicle', 'PER': 'Person', 'LOC': 'Location', 'Job-Title': 'Job-Title',
               'ORG': 'Organization',
               'GPE': 'Geopolitical-Entity', 'Time': 'Time', 'FAC': 'Facility', 'Numeric': 'Numeric',
               'WEA': 'Weapons', 'TIM': 'Time',
               'Sentence': 'Sentence', 'Crime': 'Crime', 'Contact-Info': 'Contact-Information'}


def reverse_direction(x, y, pad_token_id=1):
    input_ids = torch.cat([y['decoder_input_ids'], y['lm_labels'][:, -1:]], 1)
    attention_mask = torch.ones_like(input_ids)
    attention_mask[input_ids == pad_token_id] = 0
    decoder_input_ids = x['input_ids'][:, :-1]
    lm_labels = x['input_ids'][:, 1:]
    x = {'input_ids': input_ids, 'attention_mask': attention_mask}
    y = {'decoder_input_ids': decoder_input_ids, 'lm_labels': lm_labels}
    return x, y


class ACEDataset(Dataset):

    def __init__(
            self,
            path,
            tokenizer,
            device=torch.device('cpu'),
            use_recategorization=False,
            remove_longer_than=None,
            dereify=True,
    ):
        self.path = path
        self.tokenizer = tokenizer
        self.device = device
        # self.new_read_fun=new_read_raw_event_graph
        # sentence, event_mentions = read_raw_event_graph(path)
        sentence, event_mentions, entity_mentions = self.new_read_fun(path)
        agumented_sentence = self.get_agumented_sentence(sentence, entity_mentions)
        # e_d_result = self.read_fun()
        self.node = []
        self.sentences = []
        self.linearized = []
        self.linearized_extra = []
        self.remove_longer_than = remove_longer_than
        for g_id, g in enumerate(event_mentions):
            # 获取node_sequence,以及对应的ids
            entitys = entity_mentions[g_id]
            # e_d_list = e_d_result[g_id]
            # new_entitys=self.deal_entitys(entitys)
            node_id, node_token, node_graph = self.tokenizer.linearize(g)
            node_graph['entitys'] = entitys
            # node_graph['e_d_list'] = e_d_list
            try:
                self.tokenizer.batch_encode_sentences(agumented_sentence[g_id])
            except:
                logging.warning('Invalid sentence!')
                continue

            if remove_longer_than and len(node_id) > remove_longer_than:
                continue
            if len(node_id) > 1024:
                logging.warning('Sequence longer than 1024 included. BART does not support it!')

            self.sentences.append(agumented_sentence[g_id][0])
            self.node.append(node_token)
            self.linearized.append(node_id)
            self.linearized_extra.append(node_graph)
            # self.linearized_extra.append(entitys)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sample = {}
        sample['id'] = idx
        sample['sentences'] = self.sentences[idx]
        if self.linearized is not None:
            sample['linearized_graphs_ids'] = self.linearized[idx]
            sample.update(self.linearized_extra[idx])
        return sample

    def size(self, sample):
        return len(sample['linearized_graphs_ids'])

    def load_ace_dataset(self, path):
        assert path
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        f.close()
        return data

    def read_fun(self):
        data_path = 'get_pre_info/e_d_result.json'
        with open(data_path, encoding='utf-8') as f:
            data_lines = json.load(f)
        f.close()
        return data_lines

    def get_agumented_sentence(self, sentence, entity_mentions_list):
        entity_type_list = ['PER', 'ORG', 'GPE', 'LOC', 'Time', 'TIM', 'FAC', 'VEH', 'WEA', 'Sentence', 'Crime',
                            'Numeric', 'Job-Title']
        import re
        import string
        agumented_sentence = []
        for id, entity_mention in enumerate(entity_mentions_list):
            entity_list = []
            entity_type_entity_dict = {}
            for entity_dict in entity_mention:
                entity_type = entity_dict['entity-type'].split(':')[0]
                entity = entity_dict['head'].split()[-1]
                if entity_type in entity_type_list:
                    if entity not in entity_list:
                        entity_list.append(entity)
            word_list = sentence[id][0].split()
            new_word_list = [re.sub('[{}]'.format(string.punctuation), "", word) for word in word_list]
            # for entity in entity_list:
            #     if entity in new_word_list:
            #         need_add_index = new_word_list.index(entity)
            #         word_list.insert(need_add_index, entity_type_entity_dict[entity])
            word_list = new_word_list + ['<entity>'] + entity_list
            agumented_sentence.append([' '.join(word_list)])
        return agumented_sentence

    def deal_entitys(self, entitys):
        pass

    def new_read_fun(self, path):
        data_list = self.load_ace_dataset(path)
        sentence = []
        event_mentions = []
        entity_mentions = []
        for data_dict in data_list:
            if len(data_dict['golden_event_mentions']) == 0:
                sentence.append([data_dict['sentence']])
                event_mentions.append(data_dict['golden_event_mentions'])
                entity_mention = data_dict['golden_entity_mentions']
                for item in entity_mention:
                    text = item['text'].split()
                    head = item['head']['text'].split(' ')
                    if head[-1] in text:
                        item['head'] = head[-1]
                    else:
                        if len(text) == 1:
                            if head[-1] in item['text'].split('-'):
                                item['head'] = head[-1]
                            else:
                                item['head'] = text[-1]
                        else:
                            item['head'] = text[-1]
                entity_mentions.append(data_dict['golden_entity_mentions'])
            else:
                sentence.append([' '.join(data_dict['word']) + '.'])
                event_mentions.append(data_dict['golden_event_mentions'])
                entity_mentions.append(data_dict['golden_entity_mentions'])
        assert len(sentence) == len(event_mentions)
        return sentence, event_mentions, entity_mentions

    def collate_fn(self, samples, device=torch.device('cpu')):
        x = [s['sentences'] for s in samples]
        entitys = samples[0]['entitys']
        # e_d_list = samples[0]['e_d_list']
        x, extra = self.tokenizer.batch_encode_sentences(x, device=device)
        if 'linearized_graphs_ids' in samples[0]:
            y = [s['linearized_graphs_ids'] for s in samples]
            y, extra_y = self.tokenizer.batch_encode_graphs_from_linearized(y, samples, device=device)
            extra.update(extra_y)
        else:
            y = None
        extra['ids'] = [s['id'] for s in samples]
        return x, y, extra, entitys


class ACEDatasetTokenBatcherAndLoader:

    def __init__(self, dataset, batch_size=800, device=torch.device('cpu'), shuffle=False, sort=False):
        assert not (shuffle and sort)
        self.batch_size = batch_size
        self.tokenizer = dataset.tokenizer
        self.dataset = dataset
        self.device = device
        self.shuffle = shuffle
        self.sort = sort

    def __iter__(self):
        it = self.sampler()
        it = ([[self.dataset[s] for s in b] for b in it])
        it = (self.dataset.collate_fn(b, device=self.device) for b in it)
        return it

    @cached_property
    def sort_ids(self):
        lengths = [len(s.split()) for s in self.dataset.sentences]
        ids, _ = zip(*sorted(enumerate(lengths), reverse=True))
        ids = list(ids)
        return ids

    def sampler(self):
        ids = list(range(len(self.dataset)))[::-1]

        if self.shuffle:
            random.shuffle(ids)
        if self.sort:
            ids = self.sort_ids.copy()

        batch_longest = 0
        batch_nexamps = 0
        batch_ntokens = 0
        batch_ids = []

        def discharge():
            nonlocal batch_longest
            nonlocal batch_nexamps
            nonlocal batch_ntokens
            ret = batch_ids.copy()
            batch_longest *= 0
            batch_nexamps *= 0
            batch_ntokens *= 0
            batch_ids[:] = []
            return ret

        while ids:
            idx = ids.pop()
            size = self.dataset.size(self.dataset[idx])
            cand_batch_ntokens = max(size, batch_longest) * (batch_nexamps + 1)
            if cand_batch_ntokens > self.batch_size and batch_ids:
                yield discharge()
            batch_longest = max(batch_longest, size)
            batch_nexamps += 1
            batch_ntokens = batch_longest * batch_nexamps
            batch_ids.append(idx)

            if len(batch_ids) == 1 and batch_ntokens > self.batch_size:
                yield discharge()

        if batch_ids:
            yield discharge()
