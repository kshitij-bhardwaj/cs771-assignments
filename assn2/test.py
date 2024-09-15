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
grid_search = GridSearchCV(model, param_grid, scoring='accuracy', cv=5)  # Adjust cv for appropriate cross-validation folds
grid_search.fit(features_train, y_train)
best_estimator = grid_search.best_estimator_
best_score = grid_search.best_score_
print(f"Best score achieved: {best_score:.4f}")
print(f"Best parameters: {grid_search.best_params_}")
