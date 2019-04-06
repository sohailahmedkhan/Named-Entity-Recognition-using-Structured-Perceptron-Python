from collections import Counter
from itertools import product
from collections import defaultdict
from sklearn.metrics import f1_score
import random
import operator
import sys
import time


def load_dataset_sents(file_path, as_zip=True, to_idx=False, token_vocab=None, target_vocab=None): 
    targets = []
    inputs = []
    zip_inps = []
    with open(file_path) as f:
        for line in f:
            sent, tags = line.split('\t')
            words = [token_vocab[w.strip()] if to_idx else w.strip() for w in sent.split()] 
            ner_tags = [target_vocab[w.strip()] if to_idx else w.strip() for w in tags.split()] 
            inputs.append(words)
            targets.append(ner_tags)
            zip_inps.append(list(zip(words, ner_tags)))
    return zip_inps if as_zip else (inputs, targets)

#Get the word_label counts in the corpus
def get_current_word_current_label_counts(train_data):
    train_set = []
    counts = {}
    for i in range(len(train_data)):
        train_set.extend(train_data[i])
    counts = Counter(train_set)
    
    return counts

#Implementation for PHI1
def phi_1(words, labels, cw_cl_counts):
    dictionary = defaultdict(int)
    #Making a dictionary with word, labels and their counts
    for i in range(len(words)):
        if (words[i], labels[i]) in cw_cl_counts:
            dictionary[words[i],labels[i]] += 1
        else:
            dictionary[words[i],labels[i]] = 0
    return dictionary

#Implementation for PHI2
def phi_2(tokens, tags, features):
    dictionary = defaultdict(int)
    tags = list(tags)
    #Adding 'None' at start of every sentence as shown in the slides
    a = ['None']
    tags = a + tags
    tag_tag_features = []
    #Getting the combinations of two consecutive tags
    key2 = 1
    for i in range(len(tags)-1):
        tag_tag_features.append(tags[i] + "_" + tags[key2])
        key2 += 1

    #Making a dictionary with tag_tag and their counts
    feature_count_tags = Counter(tag_tag_features)
    for tag in tag_tag_features:
        if tag in feature_count_tags:
            dictionary[tag] += 1

    return dictionary

#Perceptron train of PHI1
def phi1_perceptron_train(train_data, features, maxIter):
    labels = ["O", "PER", "LOC", "ORG", "MISC"]
    all_possible_labels = []
    w = defaultdict(int)

    for iterr in range(maxIter):
        print("Iteration #: ", iterr+1, " for Phi1 Train")
        for sentence in train_data:
            words = []
            #Generating all possible labels
            all_possible_labels = list(product(labels,repeat = len(sentence)))
            sentence_labels = []
            #getting all words in sentence in words list
            for word, label in sentence:
                words.append(word)
                sentence_labels.append(label)

            #Predict
            maxVal = -1
            for label in all_possible_labels:
                #y = 0
                phi = phi_1(words, label, features)
                count = 0
                for key in phi:
                    count += w[key] * phi[key]

                    if count > maxVal:
                        maxVal = count
                        predict_label = list(label)
                        predict_phi = phi

            correct_phi = phi_1(words, sentence_labels, features)
            #Adjust weights
            if predict_label != sentence_labels:

                for key in correct_phi:
                    w[key] += correct_phi[key]

                for key in predict_phi:
                    w[key] -= predict_phi[key]
    return w

def phi1_perceptron_test(test_data, w, features):
    labels = ["O", "PER", "LOC", "ORG", "MISC"]
    all_possible_labels = []
    #w = defaultdict(int)
    correct = []
    predicted = []
    for sentence in test_data:
        words = []
        all_possible_labels = list(product(labels,repeat = len(sentence)))
        sentence_labels = []
        for word, label in sentence:
            words.append(word)
            sentence_labels.append(label)
        correct.append(sentence_labels)
        #Predict
        maxVal = -1
        for label in all_possible_labels:
            phi = phi_1(words, label, features)
            count = 0
            for key in phi:
                count += w[key] * phi[key]
                if count > maxVal:
                    maxVal = count
                    predict_label = list(label)
                    predict_phi = phi
        predicted.append(predict_label)

    #Flatting the lists with correct and predicted labels
    flat_cor = []
    flat_pre = []
    for sublist in correct:
        for item in sublist:
            flat_cor.append(item)
        
    for sublist in predicted:
        for item in sublist:
            flat_pre.append(item)

    return flat_cor, flat_pre

def phi1_phi2_perceptron_train(train_data, features, feature_count_tags, maxIter):
    labels = ["O", "PER", "LOC", "ORG", "MISC"]
    all_possible_labels = []
    w = defaultdict(int)
    for i in range(maxIter):
        print("Iteration #: ", i+1, " for Phi1 + Phi2 Train")
        for sentence in train_data:
            words = []
            all_possible_labels = list(product(labels,repeat = len(sentence)))
            sentence_labels = []
            for word, label in sentence:
                words.append(word)
                sentence_labels.append(label)

            #Predict
            maxVal = -1
            for label in all_possible_labels:
                phi1 = phi_1(words, label, features)
                phi2 = phi_2(words, label, feature_count_tags)
                phi = {**phi1, **phi2}
                count = 0
                for key in phi:
                    count += w[key] * phi[key]
                    
                    if count > maxVal:
                        maxVal = count
                        predict_label = list(label)
                        predict_phi = phi

            correct_phi1 = phi_1(words, sentence_labels, features)
            correct_phi2 = phi_2(words, sentence_labels, feature_count_tags)
            correct_phi = {**correct_phi1, **correct_phi2}
            if predict_label != sentence_labels:


                for key in correct_phi:
                    w[key] += correct_phi[key]

                for key in predict_phi:
                    w[key] -= predict_phi[key]
    return w

def phi1_phi2_perceptron_test(test_data, w, features, feature_count_tags):

    labels = ["O", "PER", "LOC", "ORG", "MISC"]
    all_possible_labels = []
    #w = defaultdict(int)
    correct = []
    predicted = []
    for sentence in test_data:
        words = []
        all_possible_labels = list(product(labels,repeat = len(sentence)))
        sentence_labels = []
        for word, label in sentence:
            words.append(word)
            sentence_labels.append(label)
        correct.append(sentence_labels)
        #Predict
        maxVal = -1
        for label in all_possible_labels:
            phi1 = phi_1(words, label, features)
            phi2 = phi_2(words, label, feature_count_tags)
            phi = {**phi1, **phi2}
            
            count = 0
            for key in phi:
                count += w[key] * phi[key]
            
                if count > maxVal:
                    maxVal = count
                    predict_label = list(label)
                    predict_phi = phi
                    
        predicted.append(predict_label)

    flat_cor = []
    flat_pre = []
    for sublist in correct:
        for item in sublist:
            flat_cor.append(item)
            
    for sublist in predicted:
        for item in sublist:
            flat_pre.append(item)

    return flat_cor, flat_pre

#Getting top 10 feature types
def get_top_features(w):
    labels_ORG = {}
    labels_O = {}
    labels_PER = {}
    labels_MISC = {}
    labels_LOC = {}

    for tag, weight in w.items():
        if len(tag) > 1:
            #print(tag)
            split_tag = tag[1]
            #print(split_tag, tag)
            if split_tag == 'ORG':
                labels_ORG[tag] = weight
            elif split_tag == 'O':
                labels_O[tag] = weight
            elif split_tag == 'PER':
                labels_PER[tag] = weight
            elif split_tag == 'LOC':
                labels_LOC[tag] = weight
            elif split_tag == 'MISC':
                labels_MISC[tag] = weight

    sorted_ORG = sorted(labels_ORG.items(), key=operator.itemgetter(1), reverse=True)
    sorted_O = sorted(labels_O.items(), key=operator.itemgetter(1), reverse=True)
    sorted_PER = sorted(labels_PER.items(), key=operator.itemgetter(1), reverse=True)
    sorted_LOC = sorted(labels_LOC.items(), key=operator.itemgetter(1), reverse=True)
    sorted_MISC = sorted(labels_MISC.items(), key=operator.itemgetter(1), reverse=True)

    print("\nTop 10 Features for:  LOC")
    print(sorted_LOC[:10])
    print("\nTop 10 Features for:  PER")
    print(sorted_PER[:10])
    print("\nTop 10 Features for:  MISC")
    print(sorted_MISC[:10])
    print("\nTop 10 Features for:  ORG")
    print(sorted_ORG[:10])
    print("\nTop 10 Features for:  O")
    print(sorted_O[:10])

def main():
    #Getting file paths from the command line arguments
    train_path =  sys.argv[1]
    test_path =  sys.argv[2]
    flat_cor = []
    flat_pre = []
    maxIter = 3
    
    start = time.time()
    print("\n ------------------ Working..." + "\t Please Wait... ------------------")


    train_data = load_dataset_sents(train_path)
    test_data = load_dataset_sents(test_path)
    random.seed(1)
    random.shuffle(train_data)
    random.shuffle(test_data)

    #getting word, tag counts in the corpus
    cw_cl_counts = {}
    cw_cl_counts = get_current_word_current_label_counts(train_data)

    #retaining features with counts greater than or equal to 3
    features = {}
    for key, value in cw_cl_counts.items():
        if value >= 3:
            features[key] = value

    #Getting all the possible tag_tag features in the corpus and their counts
    tag_tag_feature_counts = []
    for sentence in train_data:
        tokens, tags = zip(*sentence) 
        tags = list(tags)
        a = ['None']
        tags = a + tags
        key2 = 1
        for i in range(len(tags)-1):
            tag_tag_feature_counts.append(tags[i] + "_" + tags[key2])
            key2 += 1

    feature_count_tags = Counter(tag_tag_feature_counts)
   
   #Getting results for PHI1
    weights_phi1 = phi1_perceptron_train(train_data, cw_cl_counts, maxIter)
    flat_cor_phi1, flat_pre_phi1 = phi1_perceptron_test(test_data, weights_phi1, cw_cl_counts)
    get_top_features(weights_phi1)

    print("\n ---------------------------------------------------------------------------")
    f1_micro = f1_score(flat_cor_phi1, flat_pre_phi1, average='micro', labels=['ORG', 'MISC', 'PER', 'LOC'])
    print('F1 Score for PHI 1: ', f1_micro)
    print("--------------------------------------------------------------------------- \n")
   

   #Getting results for PHI1 + PHI2
    weights_phi1_phi2 = phi1_phi2_perceptron_train(train_data, cw_cl_counts, feature_count_tags, maxIter)
    flat_cor_phi1_phi2, flat_pre_phi1_phi2 = phi1_phi2_perceptron_test(test_data, weights_phi1_phi2, cw_cl_counts, feature_count_tags)
    get_top_features(weights_phi1_phi2)

    print("\n ---------------------------------------------------------------------------")
    f1_micro = f1_score(flat_cor_phi1_phi2, flat_pre_phi1_phi2, average='micro', labels=['ORG', 'MISC', 'PER', 'LOC'])
    print('F1 Score for PHI 1 + PHI 2: ', f1_micro)
    print("--------------------------------------------------------------------------- \n")
    end = time.time()
    print("Total Time Elapsed: ", (end - start)/60, " minutes")
if __name__ == '__main__':
    main()
