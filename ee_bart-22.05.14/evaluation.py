import json
import torch
from tqdm import tqdm
from ee_bart.tokenization_bart import ACEBartTokenizer
entity_dict = {'VEH': 'Vehicle', 'PER': 'Person', 'LOC': 'Location', 'Job-Title': 'Job-Title',
               'ORG': 'Organization',
               'GPE': 'Geopolitical-Entity', 'Time': 'Time', 'FAC': 'Facility', 'Numeric': 'Numeric',
               'WEA': 'Weapons', 'TIM': 'Time',
               'Sentence': 'Sentence', 'Crime': 'Crime', 'Contact-Info': 'Contact-Information'}
INIT = 'Ġ'
from ee_bart.dataset import reverse_direction

from ee_bart.tokenization_bart import ACEBartTokenizer
tokenizer = ACEBartTokenizer.from_pretrained(
    '../bart-large',
    'vocab/',
    dfs_linearization=True,
    bfs_linearization=False,
    use_pointer_tokens=True,
    use_entity_type=True,
    use_classification=True
)
event_ids = []
for event in tokenizer.event_types:
    g_event = tokenizer.INIT + event
    g_event_id = tokenizer.encoder[g_event]
    event_ids.append(g_event_id)
arg_indexes_ids = []
for a_index in tokenizer.arg_indexes:
    arg_indexes_id = tokenizer.encoder[tokenizer.INIT + a_index]
    arg_indexes_ids.append(arg_indexes_id)
# tokenizer = ACEBartTokenizer.from_pretrained(
#     'C:\\Users\\Sunri\\Desktop\\v1\\bart-large',
#     dfs_linearization=True,
#     bfs_linearization=False,
#     use_pointer_tokens=False,
#     use_entity_type=False)


def extract_graph_dfs(seq, event_ids, arg_ids, arg_indexes, role_ids, tokenizer):
    arg_dict = {}
    new_seq = []
    for i, token in enumerate(seq):
        token_id = tokenizer.encoder[token]
        if token_id in event_ids:
            triple = [token.replace(tokenizer.INIT, '')]
            e = token
        if token_id in arg_indexes:  # 多看一个
            if token not in arg_dict:
                arg_dict[token] = seq[i + 1]
                triple.append(seq[i + 2].replace(tokenizer.INIT, ''))
                triple.append(seq[i + 1].replace(tokenizer.INIT, ''))
                new_seq.append({len(new_seq): triple})
                triple = [e.replace(tokenizer.INIT, '')]
            else:
                should_add_token = arg_dict[token]
                triple.append(seq[i + 1].replace(tokenizer.INIT, ''))
                triple.append(should_add_token.replace(tokenizer.INIT, ''))
                new_seq.append({len(new_seq): triple})
                triple = [e.replace(tokenizer.INIT, '')]
    return [new_seq, seq]


# def get_special_ids(tokenizer):
#     event_ids = []
#     for event in tokenizer.event_types:
#         g_event = tokenizer.INIT + event
#         g_event_id = tokenizer.encoder[g_event]
#         event_ids.append(g_event_id)
#     arg_indexes_ids = []
#     for a_index in tokenizer.arg_indexes:
#         arg_indexes_id = tokenizer.encoder[tokenizer.INIT + a_index]
#         arg_indexes_ids.append(arg_indexes_id)
#     role_ids = []
#     for r in tokenizer.roles:
#         g_r = tokenizer.INIT + r
#         g_r_id = tokenizer.encoder[g_r]
#         role_ids.append(g_r_id)
#     arg_ids = []
#     for a in tokenizer.args:
#         g_a = tokenizer.INIT + a
#         g_a_id = tokenizer.encoder[g_a]
#         arg_ids.append(g_a_id)
#
#     return event_ids, arg_ids, arg_indexes_ids, role_ids


# event_ids, arg_ids, arg_indexes, role_ids = get_special_ids(tokenizer)
# seq = ["Ġ<s>",
#        "ĠTransaction:Transfer-Ownership",
#        "Ġ<Argument:0>",
#        "Ġhere",
#        "ĠPlace",
#        "Ġ<Argument:1>",
#        "ĠLtd",
#        "ĠSeller",
#        "Ġ<Argument:2>",
#        "ĠDay",
#        "ĠTime-Within",
#        "Ġ</s>"]
# graph = extract_graph_dfs(seq, event_ids, arg_ids, arg_indexes, role_ids, tokenizer)


def extract_graph_bfs(seq, event_ids, arg_ids, arg_indexes, role_ids, tokenizer):
    arg_dict = {}
    new_seq = []
    e_list = []
    stop_count = 0
    arg_dict = {}
    new_seq = []
    for i, token in enumerate(seq):
        token_id = tokenizer.encoder[token]
        if token_id in event_ids:
            e_list.append(token)
    triple = [[e.replace(tokenizer.INIT, '')] for e in e_list]
    for i, token in enumerate(seq):
        token_id = tokenizer.encoder[token]
        if token_id in role_ids:
            triple[stop_count].append(token.replace(tokenizer.INIT, ''))
        if token_id in arg_indexes:
            if token not in arg_dict:
                arg_dict[token] = seq[i + 1]
                triple[stop_count].append(seq[i + 2].replace(tokenizer.INIT, ''))
                triple[stop_count].append(seq[i + 1].replace(tokenizer.INIT, ''))
                new_seq.append({len(new_seq): triple[stop_count]})
                triple[stop_count] = [e_list[stop_count].replace(tokenizer.INIT, '')]
            else:
                should_add_token = arg_dict[token]
                triple[stop_count].append(seq[i + 1].replace(tokenizer.INIT, ''))
                triple[stop_count].append(should_add_token.replace(tokenizer.INIT, ''))
                new_seq.append({len(new_seq): triple[stop_count]})
                triple[stop_count] = [e_list[stop_count].replace(tokenizer.INIT, '')]
        if token_id in [tokenizer.stop_token_id]:
            stop_count += 1
    return [new_seq, seq]


def predict_ee_graph(
        loader, model, tokenizer, beam_size=1, tokens=None, restore_name_ops=False, return_all=False):
    shuffle_orig = loader.shuffle
    sort_orig = loader.sort

    loader.shuffle = False
    loader.sort = True

    total = len(loader.dataset)
    model.eval()
    model.amr_mode = True

    if tokens is None:
        ids = []
        tokens = []
        outputs = []
        with tqdm(total=total) as bar:
            # print(loader.batch_size)
            for x, y, extra, entitys in loader:
                ii = extra['ids']
                ids.extend(ii)
                # print(x['input_ids'].size())
                # print(x.size())
                # print(beam_size)
                with torch.no_grad():
                    out, event_ids, arg_ids, arg_indexes, role_ids = model.generate(
                        **x,
                        y=y,
                        entitys=entitys,
                        max_length=1024,
                        decoder_start_token_id=0,
                        num_beams=beam_size,
                        num_return_sequences=beam_size,
                        tokenizer=tokenizer)
                outputs.append(out)
                nseq = len(ii)
                # for i1 in range(0, out.size(0), beam_size):
                #     tokens_same_source = []
                #     tokens.append(tokens_same_source)
                #     for i2 in range(i1, i1+beam_size):
                #         tokk = out[i2].tolist()
                #         tokens_same_source.append(tokk)
                bar.update(nseq)
        # reorder
        # tokens = [tokens[i] for i in ids]
        # tokens = [t for tt in tokens for t in tt]

    graphs = []
    for i1 in range(0, len(outputs), beam_size):
        # graphs_same_source = []
        # graphs.append(graphs_same_source)
        # for i2 in range(i1, i1+beam_size):
        #     tokk = tokens[i2]
        #     graph, status, (lin, backr) = tokenizer.decode_amr(tokk, restore_name_ops=restore_name_ops)
        #     graph.status = status
        #     graph.nodes = lin
        #     graph.backreferences = backr
        #     graph.tokens = tokk
        #     graphs_same_source.append(graph)
        # graphs_same_source[:] = tuple(zip(*sorted(enumerate(graphs_same_source), key=lambda x: (x[1].status.value, x[0]))))[1]

        if tokenizer.dfs_linearization:
            graph = extract_graph_dfs(outputs[i1], event_ids, arg_ids, arg_indexes, role_ids, tokenizer)
        if tokenizer.bfs_linearization:
            graph = extract_graph_bfs(outputs[i1], event_ids, arg_ids, arg_indexes, role_ids, tokenizer)
        graphs.append(graph)
    # for gps, gg in zip(graphs, loader.dataset.graphs):
    #     for gp in gps:
    #         metadata = gg.metadata.copy()
    #         metadata['annotator'] = 'bart-amr'
    #         metadata['date'] = str(datetime.datetime.now())
    #         if 'save-date' in metadata:
    #             del metadata['save-date']
    #         gp.metadata = metadata
    #
    # loader.shuffle = shuffle_orig
    # loader.sort = sort_orig
    #
    # if not return_all:
    #     graphs = [gg[0] for gg in graphs]

    return graphs


def write_predictions(predictions_path, tokenizer, graphs):
    # pieces = [penman.encode(g) for g in graphs]
    # Path(predictions_path).write_text('\n\n'.join(graphs).replace(tokenizer.INIT, ''))

    with open(predictions_path, 'w')as f:
        json.dump(graphs, f, indent=4)
    return predictions_path


def read_fun(data_path):
    with open(data_path, encoding='utf-8') as f:
        data_lines = json.load(f)
    f.close()
    return data_lines

def count_entity_fun(p_line):
    p_entity_count = len(p_line)
    p_entity_list = []
    p_new_triple_list = []
    for i, triple_item in enumerate(p_line):
        triple = triple_item[i]
        entity = [triple[0], triple[-1]]
        p_entity_list.append(entity)
        p_new_triple_list.append(triple)
    return p_entity_list, p_new_triple_list, p_entity_count
def count_pre_e_fun(p_seq):
    Event_types = {"Movement:Transport": 1, "Personnel:Elect": 1, "Personnel:Start-Position": 1,
                   "Personnel:Nominate": 1, "Conflict:Attack": 1, "Personnel:End-Position": 1, "Life:Die": 1,
                   "Contact:Meet": 1, "Life:Marry": 1, "Contact:Phone-Write": 1, "Transaction:Transfer-Money": 1,
                   "Justice:Sue": 1, "Conflict:Demonstrate": 1, "Justice:Fine": 1, "Life:Injure": 1,
                   "Business:End-Org": 1, "Justice:Trial-Hearing": 1, "Business:Start-Org": 1, "Justice:Arrest-Jail": 1,
                   "Transaction:Transfer-Ownership": 1, "Justice:Execute": 1, "Justice:Sentence": 1, "Life:Be-Born": 1,
                   "Justice:Charge-Indict": 1, "Business:Declare-Bankruptcy": 1, "Justice:Convict": 1,
                   "Justice:Release-Parole": 1, "Justice:Pardon": 1, "Justice:Appeal": 1, "Business:Merge-Org": 1,
                   "Justice:Extradite": 1, "Life:Divorce": 1, "Justice:Acquit": 1}
    p_e_count = 0
    p_e_list = []
    for node in p_seq:
        node = node.replace(INIT, '')
        if node in Event_types:
            p_e_count += 1
            p_e_list.append(node)
    return p_e_list, p_e_count
def extract_graph_entity_dfs(seq, event_ids, arg_indexes, tokenizer):
    arg_dict = {}
    new_seq = []
    for i, token in enumerate(seq):
        token_id = tokenizer.encoder[token]
        if token_id in event_ids:
            triple = [token.replace(tokenizer.INIT, '')]
            e = token
        if token_id in arg_indexes:  # 多看一个
            if token not in arg_dict:
                arg_dict[token] = seq[i + 1]
                triple.append(seq[i + 2].replace(tokenizer.INIT, ''))
                triple.append(seq[i - 1].replace(tokenizer.INIT, ''))
                new_seq.append({len(new_seq): triple})
                triple = [e.replace(tokenizer.INIT, '')]
            else:
                should_add_token = arg_dict[token]
                triple.append(seq[i + 1].replace(tokenizer.INIT, ''))
                triple.append(seq[i - 1].replace(tokenizer.INIT, ''))
                new_seq.append({len(new_seq): triple})
                triple = [e.replace(tokenizer.INIT, '')]
    return [new_seq, seq]

def count_arg_fun(p_line):
    p_arg_count = len(p_line)
    p_arg_list = []
    p_triple_list = []
    for i, triple_item in enumerate(p_line):
        triple = triple_item[str(i)]
        args = [triple[0], triple[-1]]
        p_arg_list.append(args)
        p_triple_list.append(triple)
    return p_arg_list, p_triple_list, p_arg_count


def compute_F1(p, g):
    # print('in f1')
    e_count = 0
    e_list = []
    arg_count = 0
    pre_e_count = 0
    pre_e_list = []
    pre_arg_count = 0
    pre_entity_count = 0
    correct_e_count = 0
    correct_arg_count = 0
    correct_triple_count = 0
    correct_entity_count = 0
    correct_new_triple_count = 0
    for i, line in enumerate(g):
        e_list = []
        arg_list = []
        triple_list = []
        entity_type_list = []
        new_triple_list = []
        p_line = p[i]
        e_count += len(line['golden_event_mentions'])
        for e in line['golden_event_mentions']:
            e_list.append(e['event_type'])
            # arg_count += len(e['arguments'])
            one_event_arg_role_list = []
            one_event_entity_type_role_list = []
            for a in e['arguments']:
                head = a['head'].split()
                text = a['text'].split()
                if head[-1] in text:
                    arg = head[-1]
                else:
                    arg = text[-1]
                role = a['role']
                entity_type = entity_dict[a['entity-type'].split(':')[0]]
                entity_type_list.append([e['event_type'], entity_type])
                if [e['event_type'], role, arg] not in one_event_arg_role_list:
                    one_event_arg_role_list.append([e['event_type'], role, arg])
                    arg_list.append([e['event_type'], arg])
                    triple_list.append([e['event_type'], role, arg])
                    new_triple_list.append([e['event_type'], role, entity_type])
                    arg_count += 1
        pre_e_list, single_pre_e_count = count_pre_e_fun(p_line[1])
        pre_arg_list, pre_triple_list, single_pre_arg_count = count_arg_fun(p_line[0])
        p_entity_line = extract_graph_entity_dfs(p_line[1], event_ids, arg_indexes_ids, tokenizer)
        pre_entity_list, pre_new_triple_list, single_pre_entity_count = count_entity_fun(p_entity_line[0])
        pre_e_count += single_pre_e_count
        pre_arg_count += single_pre_arg_count
        pre_entity_count += single_pre_entity_count
        # print(arg_list)
        # print(pre_arg_list)
        # print(e_list)
        # print(pre_e_list)
        for p_e in pre_e_list:
            if p_e in e_list:
                correct_e_count += 1
                e_list.remove(p_e)
        # print(pre_e_list)
        # print(e_list)

        for p_arg in pre_arg_list:
            if p_arg in arg_list:
                correct_arg_count += 1
                arg_list.remove(p_arg)
        for p_entity in pre_entity_list:
            if p_entity in entity_type_list:
                correct_entity_count += 1
                entity_type_list.remove(p_entity)
        # print(pre_arg_list)
        # print(arg_list)
        for p_triple in pre_triple_list:
            if p_triple in triple_list:
                correct_triple_count += 1
                triple_list.remove(p_triple)
        for p_new_triple in pre_new_triple_list:
            if p_new_triple in new_triple_list:
                correct_new_triple_count += 1
                new_triple_list.remove(p_new_triple)
    # 事件检测
    print('e_num is %d' % (e_count))
    print('pre_e_num is %d' % (pre_e_count))
    print('correct_e_num is %d' % (correct_e_count))
    print('arg_num is %d' % (arg_count))
    print('pre_arg_num is %d' % (pre_arg_count))
    print('correct_arg_num is %d' % (correct_arg_count))
    print('correct_triple_num is %d' % (correct_triple_count))
    print('entity_num is %d' % (arg_count))
    print('pre_entity_num is %d' % (pre_entity_count))
    print('correct_entity_num is %d' % (correct_entity_count))
    print('correct_new_triple_num is %d' % (correct_new_triple_count))

    if correct_triple_count > correct_arg_count:
        print('wrong')
    if pre_e_count == 0:
        e_detect_p = 0.0
    else:
        e_detect_p = (correct_e_count / pre_e_count)
    if e_count == 0:
        e_detect_r = 0.0
    else:
        e_detect_r = (correct_e_count / e_count)
    if e_detect_p == 0 or e_detect_r == 0:
        e_detect_f = 0
    else:
        e_detect_f = ((e_detect_p * e_detect_r) * 2) / (e_detect_p + e_detect_r)
    # 论元检测
    if pre_arg_count == 0:
        arg_detect_p = 0
    else:
        arg_detect_p = (correct_arg_count / pre_arg_count)
    if arg_count == 0:
        arg_detect_r = 0
    else:
        arg_detect_r = (correct_arg_count / arg_count)
    if arg_detect_p == 0 or arg_detect_r == 0:
        arg_detect_f = 0
    else:
        arg_detect_f = ((arg_detect_p * arg_detect_r) * 2) / (arg_detect_p + arg_detect_r)
    # 论元分类
    if pre_arg_count == 0:
        triple_detect_p = 0
    else:
        triple_detect_p = (correct_triple_count / pre_arg_count)
    if arg_count == 0:
        triple_detect_r = 0
    else:
        triple_detect_r = (correct_triple_count / arg_count)
    if triple_detect_p == 0 or triple_detect_r == 0:
        triple_detect_f = 0
    else:
        triple_detect_f = ((triple_detect_p * triple_detect_r) * 2) / (triple_detect_p + triple_detect_r)
    # entity_type检测
    if pre_entity_count == 0:
        entity_detect_p = 0
    else:
        entity_detect_p = (correct_entity_count / pre_entity_count)
    if arg_count == 0:
        entity_detect_r = 0
    else:
        entity_detect_r = (correct_entity_count / arg_count)
    if entity_detect_p == 0 or entity_detect_r == 0:
        entity_detect_f = 0
    else:
        entity_detect_f = ((entity_detect_p * entity_detect_r) * 2) / (entity_detect_p + entity_detect_r)
    # entity_type分类
    if pre_entity_count == 0:
        entity_triple_detect_p = 0
    else:
        entity_triple_detect_p = (correct_new_triple_count / pre_entity_count)
    if arg_count == 0:
        entity_triple_detect_r = 0
    else:
        entity_triple_detect_r = (correct_new_triple_count / arg_count)
    if entity_triple_detect_p == 0 or entity_triple_detect_r == 0:
        entity_triple_detect_f = 0
    else:
        entity_triple_detect_f = ((entity_triple_detect_p * entity_triple_detect_r) * 2) / (
                entity_triple_detect_p + entity_triple_detect_r)
    score = {
        'e_detect_p': e_detect_p,
        'e_detect_r': e_detect_r,
        'e_detect_f': e_detect_f,
        'arg_detect_p': arg_detect_p,
        'arg_detect_r': arg_detect_r,
        'arg_detect_f': arg_detect_f,
        'triple_detect_p': triple_detect_p,
        'triple_detect_r': triple_detect_r,
        'triple_detect_f': triple_detect_f,
        'entity_detect_p': entity_detect_p,
        'entity_detect_r': entity_detect_r,
        'entity_detect_f': entity_detect_f,
        'entity_triple_detect_p': entity_triple_detect_p,
        'entity_triple_detect_r': entity_triple_detect_r,
        'entity_triple_detect_f': entity_triple_detect_f
    }
    return score


def compute_metric(test_path, predictions_path, extra):
    # print('in')
    gold_data_lines = read_fun(test_path)
    # print(1)
    pred_data_lines = read_fun(predictions_path)
    # print(2)
    score = compute_F1(pred_data_lines, gold_data_lines)
    # print(3)
    return score


if __name__ == "__main__":
    test_path = 'D:\\test_bart\\f_test\\f_data\\test.json'
    predictions_path = 'D:\\test_bart\\f_test\\bin\\33.json'
    compute_metric(test_path, predictions_path, 1)
