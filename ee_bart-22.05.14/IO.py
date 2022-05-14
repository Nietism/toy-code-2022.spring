import json


def load_ace_dataset(path):
    assert path
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    f.close()
    return data


def read_raw_event_graph(path):
    data_list = load_ace_dataset(path)
    sentence = []
    event_mentions = []
    for data_dict in data_list:
        sentence.append([' '.join(data_dict['word']) + '.'])
        event_mentions.append(data_dict['golden_event_mentions'])
    assert len(sentence) == len(event_mentions)
    return sentence, event_mentions


def new_read_raw_event_graph(path):
    data_list = load_ace_dataset(path)
    sentence = []
    event_mentions = []
    for data_dict in data_list:
        if len(data_dict['golden_event_mentions']) == 0:
            sentence.append([data_dict['sentence']])
            event_mentions.append(data_dict['golden_event_mentions'])
        else:
            sentence.append([' '.join(data_dict['word']) + '.'])
            event_mentions.append(data_dict['golden_event_mentions'])
    assert len(sentence) == len(event_mentions)
    return sentence, event_mentions
