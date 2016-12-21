from bs4 import BeautifulSoup
from collections import namedtuple
from collections import defaultdict
 
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import PorterStemmer

import matplotlib.pyplot as plt

import re
import collections
import csv

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import cross_validation

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
LabelDoc = namedtuple('LabelDoc', 'words tags')


def paragraph_to_words(paragraph, remove_stopwords=False, lemmatize=True, stem=False):
    words = BeautifulSoup(paragraph["review"], "html.parser").get_text()
    words = re.sub("[^a-zA-Z]", " ", words)
    # tokenizer = RegexpTokenizer(r'\w+')
    # words = tokenizer.tokenize(words.strip().lower())
    words = words.lower().split()

    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]

    if lemmatize:
        words = [lemmatizer.lemmatize(w) for w in words]

    if stem:
        words = [stemmer.stem(w) for w in words]

    return LabelDoc(words, paragraph["id"])


def build_fixed_size_dataset(input_data, vocabulary_size):
    words = []
    for i in input_data:
        words.extend(i.words)
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for tup in input_data:
        word_data = []
        for word in tup.words:
            if word in dictionary:
                index = dictionary[word]
            else:
                index = 0
                unk_count += 1
            word_data.append(index)
        data.append(LabelDoc(word_data, tup.tags))
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary


def build_dataset_with_frequent_words(input_data, min_freq):
    words = []
    for i in input_data:
        words.extend(i.words)

    count_org = collections.Counter(words).most_common()
    # count_org = defaultdict(int)
    # for doc in input_data:
    #     for word in doc.words:
    #         count_org[word] += 1

    unk_count = 0
    count = [['UNK', -1]]
    # for word, c in count_org.iteritems():
    for word, c in count_org:
        word_tuple = [word, c]
        if c >= min_freq:
            count.append(word_tuple)
        else:
            unk_count += c
    count[0][1] = unk_count

    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = []
    for tup in input_data:
        word_data = []
        for word in tup.words:
            if word in dictionary:
                index = dictionary[word]
            else:
                index = 0
            word_data.append(index)
        data.append(LabelDoc(word_data, tup.tags))
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary


def random_forest_classify(my_train_data, my_train_label, my_test_data, estimators):
    clf = RandomForestClassifier(n_estimators=estimators)
    scores = cross_validation.cross_val_score(clf, my_train_data, my_train_label, cv=5)
    print("random forest(%d) accuracy: %0.3f (+/- %0.3f)" % (estimators, scores.mean(), scores.std() * 2))
    clf.fit(my_train_data, my_train_label)
    my_test_label = clf.predict(my_test_data)
    file_name = "random_forest_%d.csv" % estimators
    save_data(my_test_label, file_name)


def gradient_boosting_classify(my_train_data, my_train_label, my_test_data, estimators):
    clf = GradientBoostingClassifier(n_estimators=estimators)
    scores = cross_validation.cross_val_score(clf, my_train_data, my_train_label, cv=5)
    print("gradient boosting(%d) accuracy: %0.3f (+/- %0.3f)" % (estimators, scores.mean(), scores.std() * 2))
    clf.fit(my_train_data, my_train_label)
    my_test_label = clf.predict(my_test_data)
    file_name = "gradient_boosting_%d.csv" % estimators
    save_data(my_test_label, file_name)


def svc_classify(my_train_data, my_train_label, my_test_data, svc_c):
    # clf = svm.SVC(C=svc_c, kernel='poly')
    clf = svm.SVC(C=svc_c)
    scores = cross_validation.cross_val_score(clf, my_train_data, my_train_label, cv=5)
    print("svc(C=%.1f) accuracy: %0.3f (+/- %0.3f)" % (svc_c, scores.mean(), scores.std() * 2))
    clf.fit(my_train_data, my_train_label)
    my_test_label = clf.predict(my_test_data)
    file_name = "svc_%.1f.csv" % svc_c
    save_data(my_test_label, file_name)


def save_data(labels, file_name):
    with open(file_name, 'wb') as my_file:
        my_writer = csv.writer(my_file)
        for label in labels:
            tmp = []
            tmp.append(label)
            my_writer.writerow(tmp)


def plot_with_para_labels(low_dim_embs, para_labels, filename='tsne-para.png'):
    assert low_dim_embs.shape[0] == len(para_labels), "label number must equal embedding number"
    plt.clf()
    plt.figure(figsize=(200, 36))
    axes = plt.axes([0.5, 0.1, 0.4, 0.8])
    axes.scatter(low_dim_embs[:, 0], low_dim_embs[:, 1], marker='o', cmap=plt.cm.seismic, s=80)
    for label, x, y in zip(para_labels, low_dim_embs[:, 0], low_dim_embs[:, 1]):
        plt.annotate(
            label,
            xy=(x, y),
            xytext=(-8, -3),
            textcoords='offset points',
            ha='right',
            va='bottom',
            bbox=None,
            arrowprops=None)
    plt.savefig(filename)


def plot_with_word_labels(low_dim_embs, labels, filename='tsne-word.png'):
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    plt.clf()
    plt.figure(figsize=(18, 18))  #in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')

    plt.savefig(filename)
