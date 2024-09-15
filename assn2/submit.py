import numpy as np
from sklearn.tree import DecisionTreeClassifier
# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py

# DO NOT PERFORM ANY FILE IO IN YOUR CODE

# DO NOT CHANGE THE NAME OF THE METHOD my_fit or my_predict BELOW
# IT WILL BE INVOKED BY THE EVALUATION SCRIPT
# CHANGING THE NAME WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here
# For example, classes to create the Tree, Nodes etc
def generate_bigrams(words):
    """Generates all bigrams (adjacent character pairs) from a list of words."""
    bigrams = set()  # Use set to avoid duplicates
    for word in words:
        bigrams.update(get_bigrams(word))
    return bigrams

def get_bigrams(word):
    """Generates all bigrams (adjacent character pairs) from a word."""
    return [word[i:i+2] for i in range(len(word)-1)]

def encode_bigram_features(word, bigrams):
    """Encodes bigram presence as binary features."""
    features = [0] * len(bigrams)
    for i, bigram in enumerate(bigrams):
        if bigram in get_bigrams(word):
            features[i] = 1
    return features
################################
# Non Editable Region Starting #
################################
def my_fit( words ):
################################
#  Non Editable Region Ending  #
################################
	
	# Do not perform any file IO in your code
	# Use this method to train your model using the word list provided
    all_bigrams = generate_bigrams(words)
    bigrams = None
    if not bigrams:
        bigrams = list(all_bigrams)

    features = []
    labels = []
    for word in words:
        encoded_features = encode_bigram_features(word, bigrams)
        features.append(encoded_features)
        labels.append(word)

    model = DecisionTreeClassifier(criterion='entropy', min_samples_split=2, min_samples_leaf=1, max_depth=None, random_state=42)
    model.fit(features, labels)
    return model, bigrams  # Return model and feature names


################################
# Non Editable Region Starting #
################################
def my_predict( model, bigram_list ):
################################
#  Non Editable Region Ending  #
################################
	
	# Do not perform any file IO in your code
	# Use this method to predict on a test bigram_list
	# Ensure that you return a list even if making a single guess
	encoded_features = encode_bigram_features("".join(bigram_list), model[1])  # Use feature names from training
	prediction = model[0].predict([encoded_features])[0]  # Predict single guess
	guess_list = [prediction]
	return guess_list  # Wrap the single prediction in a list
