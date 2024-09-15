from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
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

def my_fit(words, bigrams=None, min_samples_split=2, min_samples_leaf=1, max_depth=None):
    """Trains a decision tree model on words from a dictionary file."""
    all_bigrams = generate_bigrams(words)
    if not bigrams:
        bigrams = list(all_bigrams)

    features = []
    labels = []
    for word in words:
        encoded_features = encode_bigram_features(word, bigrams)
        features.append(encoded_features)
        labels.append(word)

    clf = DecisionTreeClassifier(min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, max_depth=max_depth)
    clf.fit(features, labels)
    return clf, bigrams  # Return model and feature names

def my_predict(model, bg):
    """Predicts a single word from a model based on a bigram list (single guess, wrapped in a list)."""
    encoded_features = encode_bigram_features("".join(bg), model[1])  # Use feature names from training
    prediction = model[0].predict([encoded_features])[0]  # Predict single guess
    return [prediction]  # Wrap the single prediction in a list

with open( "dict", 'r' ) as f:
	words = f.read().split( '\n' )[:-1]		# Omit the last line since it is empty
	num_words = len( words )


X_train, X_test, y_train, y_test = train_test_split(words, words, test_size=0.2, random_state=42)  # Replace with your data split logic
all_bigrams = generate_bigrams(X_train)
bigrams = list(all_bigrams)

features_train = []
for word in X_train:
  encoded_features = encode_bigram_features(word, bigrams)
  features_train.append(encoded_features)

# You can experiment with different splitting criteria (e.g., 'gini', 'entropy')
splitting_criteria = ['gini', 'entropy']

# Explore different values for hyperparameters based on your dataset size and complexity
param_grid = {
    'criterion': splitting_criteria,
    'min_samples_split': [2, 4, 8],
    'min_samples_leaf': [1, 2, 4]
}

model = DecisionTreeClassifier()
grid_search = GridSearchCV(model, param_grid, scoring='accuracy', cv=3)  # Adjust cv for appropriate cross-validation folds
grid_search.fit(features_train, y_train)
best_estimator = grid_search.best_estimator_
best_score = grid_search.best_score_
print(f"Best score achieved: {best_score:.4f}")
print(f"Best parameters: {grid_search.best_params_}")
