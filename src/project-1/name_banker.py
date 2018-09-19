import numpy as np

class NameBanker:
    def __init__(self, model):
        self.model = model

    # We have made the assumption that this data is for actual given loans,
    # and that the good- or bad-label are put on the customer after the event
    # of giving the loan is paid back, or not.
    def fit(self, X, y):
        self.data = [X, y]
        self.model.fit(self.data[0], self.data[1])

    # set the interest rate
    def set_interest_rate(self, rate):
        self.rate = rate

    # Predict the probability of failure for a specific person with data x
    def predict_proba(self, x):
        x = x.values.reshape(1, -1) #reshaping x
        return self.model.predict_proba(x)[0]

    # THe expected utility of granting the loan or not. Here there are two actions:
    # action = 0 do not grant the loan
    # action = 1 grant the loan
    #
    # Make sure that you extract the length_of_loan from the
    # 2nd attribute of x. Then the return if the loan is paid off to you is
    # amount_of_loan*(1 + rate)^length_of_loan.
    # The return if the loan is not paid off is -amount_of_loan.
    def expected_utility(self, x, action):
        duration = x['duration']; amount = x['amount']
        good_pred, bad_pred = self.predict_proba(x)

        utility = 0

        if action == 1:
            utility -= amount*bad_pred
            utility += amount*(pow(1 + self.rate, duration) - 1)*good_pred
        return utility

    # Return the best action. This is normally the one that maximises expected utility.
    # However, you are allowed to deviate from this if you can justify the reason.
    def get_best_action(self, x):
        return np.argmax([self.expected_utility(x, 0), self.expected_utility(x, 1)])
