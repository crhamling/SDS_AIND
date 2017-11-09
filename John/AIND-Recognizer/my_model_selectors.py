import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

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

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
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
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # Implement model selection based on BIC scores
        bic_scores = []
        # Range of values for iterating components
        num_range_components = range(self.min_n_components, self.max_n_components + 1)

        try: # Handle case where log_liklihood does not sum to 1
            # For each component get the number of data points
            for num_data_points in num_range_components:
                # Get the base model for the specified number of data points
                model = self.base_model(num_data_points)
                # Store log of the liklihood of the fitted model
                log_liklihood = model.score(self.X, self.lengths)
                # Store number of parameters
                parameters = (num_data_points ** 2) + (2 * num_data_points * model.n_features) - 1
                # Calculate BIC score
                bic_score = (-2 * log_liklihood) + (parameters * math.log(num_data_points))
                # Add the BIC score to the list
                bic_scores.append(bic_score)
        except Exception as e:
            pass

        # Use constant value for states that bic_scores were not computed
        states = num_range_components[np.argmax(bic_scores)] if bic_scores else self.n_constant
        return self.base_model(states)


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # Implement model selection based on DIC scores
        dic_scores = []
        # Range of values for iterating components
        num_range_components = range(self.min_n_components, self.max_n_components + 1)
        # Store log of the liklihoods
        log_liklihoods = []

        try: # Handle case where log_liklihood does not sum to 1
            # For each component get the number of data points
            for num_data_points in num_range_components:
                # Get the base model for the specified number of data points
                model = self.base_model(num_data_points)
                # Store log of the liklihood of the fitted model
                log_liklihoods.append(model.score(self.X, self.lengths))
            # Store sum of the log of the liklihoods
            sum_log_liklihoods = sum(logs_l)
            num_components = len(num_range_components)
            for log_liklihood in log_liklihoods:
                # Calculate DIC score
                other_words_likelihood = (sum_log_liklihoods - log_liklihood) / (num_components - 1)
                # Add the DIC score to the list
                dic_scores.append(log_liklihood - other_words_likelihood)
        except Exception as e:
            pass

        # Use constant value for states that dic_scores were not computed
        states = num_range_components[np.argmax(dic_scores)] if dic_scores else self.n_constant
        return self.base_model(states)


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # Implement model selection using CV
        mean_scores = []
        # Range of values for iterating components
        num_range_components = range(self.min_n_components, self.max_n_components + 1)

        try: # Handle case where fold_score does not sum to 1
            # For each component get the number of data points
            for num_data_points in num_range_components:
                # Get the base model for the specified number of data points
                model = self.base_model(num_data_points)
                # Store the fold scores
                fold_scores = []
                # For test indicies in fold K 
                for _, test_idx in KFold().split(self.sequences):
                    # Get test sequences
                    test_x, test_length = combine_sequences(test_idx, self.sequences)
                    # Add CV score to the list
                    fold_scores.append(model.score(test_X, test_length))

                # Calculate the mean of all fold scores
                mean_scores.append(np.mean(fold_scores))
        except Exception as e:
            pass

        # Use constant value for states that mean_scores were not computed
        states = num_range_components[np.argmax(mean_scores)] if mean_scores else self.n_constant
        return self.base_model(states)
