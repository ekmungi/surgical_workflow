import numpy as np


class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                            best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                            best * min_delta / 100)




def test():

    early_stopping = EarlyStopping(mode='min', min_delta=0.1, patience=10)
    n_bad_epochs = 0
    # list_val = list(range(1000))
    for i in range(100):
        x = np.random.choice(1000)
        if (early_stopping.step(x)):
            print("Early stop after {0} iterations".format(i))
            return
        elif n_bad_epochs > early_stopping.num_bad_epochs:
            print("ITERATION: {3} Number of bad epochs: {0}, Current: {1}, Best: {2}     <===".format(early_stopping.num_bad_epochs, 
                                                                                x, early_stopping.best, i))
        else:
            print("ITERATION: {3} Number of bad epochs: {0}, Current: {1}, Best: {2}".format(early_stopping.num_bad_epochs, 
                                                                                x, early_stopping.best, i))

        n_bad_epochs = early_stopping.num_bad_epochs

    return



if __name__ == "__main__":
    test()