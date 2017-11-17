"""
Created on Fri Nov  5 17:03:36 2017

@author: apoorv purwar (ap3644)
"""

#==============================================================================
# Initial Scores (k=1 and c=1):
#    
# test.txt - 

# Precision:0.9491525423728814 
# Recall:0.8888888888888888 
# F-Score:0.9180327868852458 
# Accuracy:0.9820466786355476
#
# dev.txt -
# Precision:0.9508196721311475 
# Recall:0.8656716417910447 
# F-Score:0.9062499999999999 
# Accuracy:0.9784560143626571

#==============================================================================
# After Parameter Tuning  (k=3 and c=0.05):
#    
# test.txt - 
# Precision:0.9655172413793104 
# Recall:0.8888888888888888 
# F-Score:0.9256198347107438 
# Accuracy:0.9838420107719928
#
# dev.txt -
# Precision:0.9649122807017544 
# Recall:0.8805970149253731 
# F-Score:0.9218749999999999 
# Accuracy:0.9820466786355476
# 
#==============================================================================
import sys
import string
import codecs
import math
import re

# Takes the content of a signle SMS as input
def extract_words(text):
    text = text.lower()  
    text.strip()                        
    t = str.maketrans("", "", string.punctuation)
    return (text.translate(t)).split()


class NbClassifier(object):

    def __init__(self, training_filename, stopword_file = "stopwords_mini.txt"):
        self.attribute_types = set()
        self.label_prior = {}    
        self.word_given_label = {} 
        self.stop_words = set()
        self.vocab = {}

        self.read_stop_words(stopword_file)
        self.collect_attribute_types(training_filename, 2)
        self.train(training_filename)   
        
    def read_stop_words(self, stopword_file):
        f = codecs.open(stopword_file, 'r', 'UTF-8')
        for line in f:
            line = line.strip()
            self.stop_words.add(line)
            
    # Compute a vocabulary consisting of the set of unique words 
    # occurring at least  k times in the training data
    def collect_attribute_types(self, training_filename, k):
        
        f = codecs.open(training_filename, 'r', 'UTF-8')
        for line in f: 
            label, text = line.split('\t', 1)
            extracted_words = extract_words(text)
            
            for word in extracted_words:
                if (word in self.vocab) and (word not in self.stop_words):
                    self.vocab[word] += 1
                elif (word not in self.vocab) and (word not in self.stop_words):
                    self.vocab[word] = 1   
        
        f.close()
        # Create the vocabulary of words from the training data  
        for word in self.vocab:
            if self.vocab[word] >= k:
                self.attribute_types.add(word)
       

    def train(self, training_filename):
        f = codecs.open(training_filename, 'r', 'UTF-8')
        word_count = {}
        line_count = 0
        for line in f:
            line_count += 1
            label, text = line.split('\t', 1)
            extracted_words = extract_words(text)
            if label in self.label_prior:
                self.label_prior[label] += 1
            else:
                self.label_prior[label] = 1
            if label in word_count:
                word_count[label] += len(extracted_words)
            else:
                word_count[label] = len(extracted_words)
                
            for word in extracted_words:
                if word in self.attribute_types:
                    if (word, label) in self.word_given_label:
                        self.word_given_label[(word, label)] += 1
                    else:
                        self.word_given_label[(word, label)] = 1
                        
        c = 0.05 # Parameter to tune missing values
        
        
        for label in self.label_prior:
            self.label_prior[label] = self.label_prior[label] / line_count
        
        #Calculate conditional probability for each word
        for word in self.attribute_types:
            for label in self.label_prior:
                if (word, label) in self.word_given_label:
                    self.word_given_label[(word, label)] = (self.word_given_label[(word, label)] + c) / (word_count[label] + c*len(self.attribute_types))
                else:
                    self.word_given_label[(word, label)] = c / (word_count[label] + c * len(self.attribute_types))
        f.close()
        
        
    def predict(self, text):
        prediction_value = {}
        extracted_words = extract_words(text)
        for label in self.label_prior:
            prediction_value[label] = math.log(self.label_prior[label])
            for word in extracted_words:
                if word in self.attribute_types:
                    prediction_value[label] += math.log(self.word_given_label[(word,label)])
        return prediction_value

    def evaluate(self, test_filename):
        
        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0
        
        f = codecs.open(test_filename, 'r', 'UTF-8')
        
        for line in f: 
            line = str(line.strip())
            content = re.split(r'\t+', line)
            prediction = self.predict(content[1])
            if prediction.get('ham') <= prediction.get('spam'):
                if content[0] == 'spam':
                    true_positive = true_positive + 1
                else:
                    false_positive = false_positive + 1
            else:
                if content[0] == 'spam':
                    false_negative = false_negative + 1
                else:
                    true_negative = true_negative + 1
                    
        precision = true_positive/(true_positive + false_positive)
        recall = true_positive/(true_positive + false_negative)
        fscore = (2 * precision * recall)/(precision + recall)
        accuracy = (true_positive + true_negative)/(true_positive + true_negative + false_positive + false_negative)
        f.close()
        
        return precision, recall, fscore, accuracy


def print_result(result):
    print("Precision:{} Recall:{} F-Score:{} Accuracy:{}".format(*result))


if __name__ == "__main__":
    
    classifier = NbClassifier(sys.argv[1])
    result = classifier.evaluate(sys.argv[2])
    print_result(result)
