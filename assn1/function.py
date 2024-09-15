import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from scipy.linalg import khatri_rao
# You are allowed to import any submodules of sklearn that learn linear models e.g. sklearn.svm etc
# You are not allowed to use other libraries such as keras, tensorflow etc
# You are not allowed to use any scipy routine other than khatri_rao

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py

# DO NOT CHANGE THE NAME OF THE METHODS my_fit, my_map etc BELOW
# THESE WILL BE INVOKED BY THE EVALUATION SCRIPT. CHANGING THESE NAMES WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here
# For example, functions to calculate next coordinate or step length
################################
# Non Editable Region Starting #
################################
def my_fit1( X_train, y0_train, y1_train, loss_function ):
################################
#  Non Editable Region Ending  #
################################

	# Train linear models for Response0 and Response1
	model0 = LinearSVC(max_iter=1000, loss=loss_function)
	model0.fit((my_map(X_train)), y0_train)
	
	W0 = model0.coef_.reshape(-1)
	b0 = model0.intercept_[0]
	model1 = LinearSVC(max_iter=1000, loss=loss_function)
	model1.fit((my_map(X_train)), y1_train)
	W1 = model1.coef_.reshape(-1)
	b1 = model1.intercept_[0]
	return W0, b0, W1, b1
	# Use this method to train your models using training CRPs
	# X_train has 32 columns containing the challenge bits
	# y0_train contains the values for Response0
	# y1_train contains the values for Response1
	
	# THE RETURNED MODELS SHOULD BE TWO VECTORS AND TWO BIAS TERMS
	# If you do not wish to use a bias term, set it to 0

################################
# Non Editable Region Starting #
################################
def my_map( X ):
################################
#  Non Editable Region Ending  #
################################
	D = 32 * 32
	# Use this method to create features.
	# It is likely that my_fit will internally call my_map to create features for train points
	return khatri_rao(X.T, X.T).T
