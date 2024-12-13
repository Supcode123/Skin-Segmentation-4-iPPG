class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0, verbose=False):
        """
        :param patience: the number of times the validation loss is not improved
        :param min_delta: minimum change to improve validation loss
        :param verbose: whether to print detailed information
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, current_score):
        """
        monitor the current validation score and decide whether to stop training

        :param current_score: the loss score of the current validation set
        """
        if self.best_score is None:
            self.best_score = current_score
        else:
            improvement = self.best_score - current_score
            if improvement > self.min_delta:
                self.best_score = current_score
                self.counter = 0  # reset the counter
                if self.verbose:
                    print(f"Validation improved to {self.best_score:.4f}")
            else:
                self.counter += 1
                if self.verbose:
                    print(f"No improvement for {self.counter} epochs")
                if self.counter >= self.patience:
                    self.early_stop = True
                    if self.verbose:
                        print("Early stopping triggered")