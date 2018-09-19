import pandas
import numpy as np
## Set up for dataset
features = ['checking account balance', 'duration', 'credit history',
            'purpose', 'amount', 'savings', 'employment', 'installment',
            'marital status', 'other debtors', 'residence time',
            'property', 'age', 'other installments', 'housing', 'credits',
            'job', 'persons', 'phone', 'foreign']
target = 'repaid'
df = pandas.read_csv('../../data/credit/german.data', sep=' ',
                     names=features+[target])
import matplotlib.pyplot as plt
numerical_features = ['duration', 'age', 'residence time', 'installment', 'amount', 'duration', 'persons', 'credits']
quantitative_features = list(filter(lambda x: x not in numerical_features, features))
X = pandas.get_dummies(df, columns=quantitative_features, drop_first=True)
encoded_features = list(filter(lambda x: x != target, X.columns))

## Test function
def test_decision_maker(X_test, y_test, interest_rate, decision_maker):
    n_test_examples = len(X_test)
    utility = 0
    ## Example test function - this is not an unbiased test as it uses the training data directly. Adapt as necessary
    for t in range(n_test_examples):
        action = decision_maker.get_best_action(X_test.iloc[t])

        good_loan = y_test.iloc[t] # assume the labels are correct
        duration = X_test['duration'].iloc[t]
        amount = X_test['amount'].iloc[t]
        # If we don't grant the loan then nothing happens
        if (action==1):
            if (good_loan == 2):
                utility -= amount
            else:
                utility += amount*(pow(1 + interest_rate, duration) - 1)
    return utility


## Main code


### Setup model
#import logistic_banker
#decision_maker = logistic_banker.LogisticBanker()
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(16, 4, 2), random_state=1)
bagging = BaggingClassifier(KNeighborsClassifier(), n_estimators=10)
knn = KNeighborsClassifier()
logistic = linear_model.LogisticRegression()
import name_banker
import random_banker

banker_logistic = name_banker.NameBanker(logistic)
banker_random = random_banker.RandomBanker()
bankers = [banker_logistic, banker_random]

from sklearn.model_selection import train_test_split
n_tests = 10
utility = 0
interest_rate = 0.005
for banker in bankers:
    for iter in range(n_tests):
        X_train, X_test, y_train, y_test = train_test_split(X[encoded_features], X[target], test_size=0.2)
        banker.set_interest_rate(interest_rate)
        banker.fit(X_train, y_train)
        utility += test_decision_maker(X_test, y_test, interest_rate, banker)

    print(utility/n_tests)
