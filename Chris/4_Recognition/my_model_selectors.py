import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose
        self.n_components = range(self.min_n_components, self.max_n_components + 1)
        
    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):

    def select(self):

        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):

    def select(self):
        
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        bic_scores = []
        try:
            for n in self.n_components:
                    model = self.base_model(n)
                    bic_score = (-2 * model.score(self.X, self.lengths)) + \
                    ((n ** 2) + (2 * n * model.n_features) - 1 * math.log(n))
                    bic_scores.append(bic_score)
        except:
            if self.verbose:
                print("failure in SelectorBIC")
                
        s = self.n_components[np.argmax(bic_scores)] if bic_scores else self.n_constant
        
        return self.base_model(s)

class SelectorDIC(ModelSelector):
        
    def select(self):
        
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        
        dic_scores = []
        log_ls = []
        try:
            for comp in self.n_components:
                model = self.base_model(comp)
                logs_l.append(model.score(self.X, self.lengths))
            sum_logs = sum(logs_l)
            length = len(self.n_components)

            for log_l in logs_l:
                other_words_likelihood = (sum_logs - log_l) / (length - 1)
                dic_scores.append(log_l - other_words_likelihood)
        except:
            if self.verbose:
                print("failure in SelectorDIC")
                
        s = self.n_components[np.argmax(dic_scores)] if dic_scores else self.n_constant

        return self.base_model(s)


class SelectorCV(ModelSelector):

    def select(self):
        
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        mean_scores = []
        split_method = KFold()
        
        try:
            for n in self.n_components:
                model = self.base_model(n)
                fold_scores = []

                for t, test_idx in split_method.split(self.sequences):
                    test_X, test_length = combine_sequences(test_idx, self.sequences)
                    fold_scores.append(model.score(test_X, test_length))
                mean_scores.append(np.mean(fold_scores))
        except:
            if self.verbose:
                print("failure in SelectorCV")
                
        s = self.n_components[np.argmax(mean_scores)] if mean_scores else self.n_constant
        
        return self.base_model(s)
