import re

TRAIN_CORPUS_PATH = "data/199801-raw.txt"
POS2TAG = {'t': 'T', 'nr': 'PER', 'ns': 'ORG', 'nt': 'LOC'}

def data_processing(path):

    file = open(path, 'r')
    corpus_pre_process = file.readlines()
    corpus_post_process = []
    for line in corpus_pre_process:
        sentence = line.strip()
        sentence = full2half(sentence)
        sentence = sentence.split()
        sentence = combine_first_second_name(sentence)
        sentence = combine_large_token(sentence)
        sentence = combine_time(sentence)
        # print(sentence)
        # print('----------')
        sentence = ' '.join(sentence[1:])
        # print(sentence)
        # print('##########')
        if sentence.strip():
            corpus_post_process.append(sentence.strip().split())
    return corpus_post_process


def full2half(sentence):
    # change full-width characters to half-width characters
    """
    full-width: 65281-65374/0xff01-0xff5e
                blank 12288/0x3000
    half-width: 33-126/0x21-0x7e
                blank 32/0x20
    """
    sentence_new = ""
    for char in sentence:
        unicode = ord(char)
        if unicode == 12288:
            unicode = 32
        elif 65281 <= unicode <= 65374:
            unicode -= 65248
        char_new = chr(unicode)
        sentence_new += char_new
    return sentence_new


def combine_first_second_name(sentence):
    # combine the first name and last name separated by corpus
    # change 温/nr  家宝/nr to 温家宝/nr
    sentence_new = []
    length = len(sentence)
    index = 0
    while index < length - 1:
        word = sentence[index]
        next_word = sentence[index + 1]
        if '/nr' in word and '/nr' in next_word:
            name = word.replace('/nr', '') + next_word
            sentence_new.append(name)
            index += 2
        else:
            sentence_new.append(word)
            index += 1
        if index == length - 1:
            sentence_new.append(sentence[index])
    return sentence_new


def combine_large_token(sentence):
    # combine the tokens inside "[" and "]"
    # change [中国/ns  国际/n  广播/vn  电台/n]nt to 中国国际广播电台/nt
    sentence_new = []
    length = len(sentence)
    index = 0
    large_token = ''
    while index < length:
        word = sentence[index]
        if '[' in word:
            word = re.sub('[a-zA-Z/\[]', '', word)
            large_token += word
        elif ']' in word:
            words = word.split(']')
            word = re.sub('[a-zA-Z/]', '', words[0]) + '/' + words[1]
            large_token += word
            sentence_new.append(large_token)
            large_token = ''
        elif large_token:
            word = re.sub('[a-zA-Z/]', '', word)
            large_token += word
        else:
            sentence_new.append(word)
        index += 1
    return sentence_new


def combine_time(sentence):
    # combine separated time
    # change 一九九七年/t  十二月/t  三十一日/t to 一九九七年十二月三十一日/t
    sentence_new = []
    length = len(sentence)
    index = 0
    time = ''
    while index < length:
        word = sentence[index]
        if '/t' in word:
            time += word.replace('/t', '')
        elif time:
            time += '/t'
            sentence_new.append(time)
            time = ''
            sentence_new.append(word)
        else:
            sentence_new.append(word)
        index += 1
    return sentence_new


def pos2tag(postion):
    # change postion to tag
    tag = POS2TAG.get(postion, 'O')
    return tag


def get_BIO(tag, index):
    # change tag to BIO tag
    if index == 0 and tag != 'O':
        tag = 'B_' + tag
    elif tag != 'O':
        tag = 'I_' + tag
    return tag


def simplify_postion_info(pos): 
    """去除词性携带的标签先验知识"""
    # change nr/ns/nt to n
    if pos in POS2TAG.keys() and pos != 't':
        pos = 'n'
    return pos


def init_sequence(sentences):
    # initialize word/position/tag sequence
    """
    use tri-gram here
    change word to character, as well as its position and tag
    placeholder is required
    """
    word_seqs = []
    position_seqs = []
    tag_seqs = []
    for sentence in sentences:
        word_seq = ['<BOS>']
        position_seq = ['un']
        tag_seq = []
        for word_with_position in sentence:
            
            word = word_with_position.split('/')[0]
            position = word_with_position.split('/')[1]
            position_seq.append(simplify_postion_info(position))

            tag = pos2tag(position)

            for i in range(len(word)):
                word_seq.append(word[i])
                tag_BIO = get_BIO(tag, i)
                tag_seq.append(tag_BIO)
        word_seq.append('<EOS>')
        word_seqs.append(word_seq)

        position_seq.append('un')
        position_seqs.append(position_seq)

        tag_seqs.append(tag_seq)
    return word_seqs, position_seqs, tag_seqs


def get_tri_gram(word_seq):
    # separate the sentence into 3-grams
    """
    change "我在睡觉" to "我在睡" "在睡觉"
    """
    window = 3
    length = len(word_seq)
    tri_gram = []
    for i in range(length - 2):
        tri_gram.append(word_seq[i:i + window])
    return tri_gram


def get_feature(tri_gram):

    tri_gram_feature = []
    for gram in tri_gram:
        feature = {
            # unigram features
            'w-1': gram[0], 
            'w': gram[1], 
            'w+1': gram[2],

            # bigram features
            'w-1_w': gram[0] + gram[1], 
            'w_w+1': gram[1] + gram[2], 

            'bias': 1.0
        }
        tri_gram_feature.append(feature)
    return tri_gram_feature


def get_model_input(word_seqs):
    tri_grams_feature = []
    for word_seq in word_seqs:
        tri_gram = get_tri_gram(word_seq)
        tri_gram_feature = get_feature(tri_gram)
        tri_grams_feature.append(tri_gram_feature)
    return tri_grams_feature


if __name__ == '__main__':
    corpus = data_processing(TRAIN_CORPUS_PATH)
    print(corpus)
