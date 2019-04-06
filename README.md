# Named-Entity-Recognition-using-Structured-Perceptron-Python

The goal of this script is to learn a named entity recogniser (NER) using the structured perceptron. 
For each word in a sequence, the named entity recogniser should predict one of the following labels:

• O: not part of a named entity
• PER: part of a person’s name
• LOC: part of a location’s name
• ORG: part of an organisation’s name
• MISC: part of a name of a different type (miscellaneous, e.g. not person, location or organisation)

# Running the Script

To run the script in Command Line use : python3 ner.py train.txt test.txt

The approach uses two different Feature Extraction techniques. 

1) Current word-current label
2) Previous label-current label

I used structurd perceptron for training.

The results are quite understandable. Also, by using more iterations better results can be anticipated. 
