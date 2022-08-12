import sklearn_crfsuite
import utils
from sklearn.model_selection import train_test_split
import pickle
from sklearn_crfsuite import metrics

TRAIN_CORPUS_PATH = "data/199801-raw.txt"
MODEL_PATH = 'model.pkl'


def get_corpus(path):
    corpus_post_processing = utils.data_processing(path)
    sequences = utils.init_sequence(corpus_post_processing)
    word_seqs = sequences[0]
    position_seqs = sequences[1]
    tag_seqs = sequences[2]
    model_input = utils.get_model_input(word_seqs)
    x_train, x_test, y_train, y_test = train_test_split(model_input, tag_seqs, test_size=0.3)
    return x_train, x_test, y_train, y_test


def train_model(x_train, y_train):
    # get thr CRF model
    model = sklearn_crfsuite.CRF(algorithm='lbfgs', c1=0.1, c2=0.2, max_iterations=100, all_possible_transitions=True)
    model.fit(x_train, y_train)
    # since label 'O' is the most common label,
    # it will make the result look better than the real one
    # then we will remove label 'O'
    labels = list(model.classes_)
    labels.remove('O')
    model_filpath = open(MODEL_PATH, 'wb')
    pickle.dump(model, model_filpath)
    return model, labels


def test_model(model_path, labels, x_test, y_test):
    # test the model
    model_filepath = open(model_path, 'rb')
    model = pickle.load(model_filepath)
    y_predict = model.predict(x_test)
    # f1_score = metrics.f1_score(y_true=y_test, y_pred=y_predict, labels=labels)
    # sort the labels according to alphabet
    sorted_labels = sorted(labels, key=lambda BIO_tag: (BIO_tag[1:], BIO_tag[0]))
    report = metrics.flat_classification_report(y_true=y_test, y_pred=y_predict, labels=sorted_labels, digits=3)
    # support: number of times each label shows up
    return report


def predict(model_path, sentence):
    # tag a new sentence
    model_filepath = open(model_path, 'rb')
    model = pickle.load(model_filepath)
    # process the raw sentence
    sentence = utils.full2half(sentence)
    word_seq = ['<BOS>']
    for char in sentence:
        word_seq.append(char)
    word_seq.append('<EOS>')
    tri_gram = utils.get_tri_gram(word_seq)
    word_seq = word_seq[1:-1]
    features = utils.get_feature(tri_gram)
    features = [features]
    # start tagging
    y_predict = model.predict(features)[0]
    # extract named entities
    named_entity = []
    if y_predict[0] != 'O':
        temp_entity = word_seq[0]
    else:
        temp_entity = ''
    for i in range(1, len(y_predict)):
        tag = y_predict[i]
        tag_previous = y_predict[i - 1]
        char = word_seq[i]
        if tag[-1] == tag_previous[-1] and tag != 'O':
            temp_entity += char
        elif tag != 'O':
            if temp_entity != '':
                named_entity.append(temp_entity)
            temp_entity = char
    if temp_entity != '':
        named_entity.append(temp_entity)
    # print(word_seq)
    # print(y_predict)
    return named_entity


def main():
    # for test model
    x_train, x_test, y_train, y_test = get_corpus(TRAIN_CORPUS_PATH)
    model, labels = train_model(x_train, y_train)
    report = test_model(MODEL_PATH, labels, x_test, y_test)
    print(report)

    # for tag a new sentence
    # sentence = '新华社十月电，毛泽东同志宣布新中国成立。'
    # sentence = sentence.replace(" ", "")
    # named_entity = predict(MODEL_PATH, sentence)
    # print(sentence)
    # print(named_entity)


if __name__ == '__main__':
    main()


