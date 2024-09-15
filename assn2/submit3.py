from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

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

def my_fit(words, features_fn=generate_bigrams, max_depth=None, classifier_type='decision_tree', **classifier_kwargs):
    """Trains a model on words from a dictionary file.

    Args:
        filename: Path to the dictionary file.
        features_fn: Function to generate features (default: bigrams).
        max_depth: Maximum depth of the decision tree (if applicable).
        classifier_type: Either 'decision_tree' or 'random_forest'.
        **classifier_kwargs: Additional arguments passed to the classifier (e.g., min_samples_split, min_samples_leaf).
    """
    all_bigrams = features_fn(words)
    features = []
    labels = []
    for word in words:
        encoded_features = encode_bigram_features(word, all_bigrams)
        features.append(encoded_features)
        labels.append(word)

    if classifier_type == 'decision_tree':
        clf = DecisionTreeClassifier(max_depth=max_depth, **classifier_kwargs)
    elif classifier_type == 'random_forest':
        clf = RandomForestClassifier(**classifier_kwargs)
    else:
        raise ValueError("Invalid classifier type. Choose 'decision_tree' or 'random_forest'.")

    clf.fit(features, labels)
    return clf, all_bigrams  # Return model and feature names

def my_predict(model, bg, threshold=0.7, num_guesses=5):
    """Predicts a single word from a model based on a bigram list.

    Args:
        model: Trained model from my_fit.
        bg: List of bigrams.
        threshold: Minimum probability for a guess to be included (default: 0.7).
        num_guesses: Maximum number of guesses to return (default: 5).
    """
    encoded_features = encode_bigram_features("".join(bg), model[1])  # Use feature names from training
    predictions, probabilities = zip(*model[0].predict_proba([encoded_features] * num_guesses))

    # Sort predictions and probabilities together based on probabilities (descending)
    sorted_results = sorted(zip(predictions, probabilities), key=lambda x: x[1], reverse=True)

    # Filter predictions based on threshold
    filtered_predictions = [p for p, prob in sorted_results if prob >= threshold]

    # Return single guess or list based on filtered predictions
    if len(filtered_predictions) == 1:
        return filtered_predictions[0][0]  # Single guess with highest probability
    elif filtered_predictions:
        return [p for p, _ in filtered_predictions[:num_guesses]]  # Top guesses within threshold
    else:
        return []  # No predictions above threshold
