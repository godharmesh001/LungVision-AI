"""
Project : LungVision-AI
Module  : Early Stopping
"""


class EarlyStopping:

    def __init__(self, patience=5):

        self.patience = patience
        self.best_score = 0.0
        self.counter = 0
        self.stop = False

    def __call__(self, score):

        if score > self.best_score:

            self.best_score = score
            self.counter = 0

        else:

            self.counter += 1

            if self.counter >= self.patience:

                self.stop = True

        return self.stop