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
        
        # By defining the n_components variable here, it lowers word error rate to less than 60%, i.e. WER < 0.60 
        self.n_components = range(self.min_n_components, self.max_n_components + 1)

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


#
# CLASS: SelectorConstant
#
class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant
    
    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)



#
# CLASS: SelectorBIC
#
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

        # TODO implement model selection based on BIC scores
        # raise NotImplementedError

        bic_scores = []  # BIC Score list
        
        # Use try/catch in case of exception
        try:
            # Iterate through all N_Components
            for n in self.n_components:
                #*******************************************
                # Quick Info:
                # BIC = −2 log L + p log N
                # L = is the likelihood of the fitted model
                # p = is the number of parameters
                # N = is the number of data points
                #*******************************************
                model = self.base_model(n)                     # Start w/the Base Model
                log_l = model.score(self.X, self.lengths)      # LOG_L
                p = n ** 2 + 2 * n * model.n_features - 1      # P(arameters)
                bic_score = -2 * log_l + p * math.log(n)       # BIC_SCORE
                bic_scores.append(bic_score)                   
                
        except Exception as e:
            pass                                               # Pass on an exceptions

        # Set the States to max BIC Score found else constant.
        states = self.n_components[np.argmax(bic_scores)] if bic_scores else self.n_constant
        
        return self.base_model(states)
        

#
# CLASS: SelectorDIC
#
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

        # TODO implement model selection based on DIC scores
        # raise NotImplementedError

        dic_scores = []    # DIC Score list
        logs_l = []        # LOGS_L list
         
        # Use try/catch in case of exception
        try:
            # Iterate through all N_Components
            for n_component in self.n_components:
                model = self.base_model(n_component)                  # Start w/the Base Model
                logs_l.append(model.score(self.X, self.lengths))      # LOG_L
            sum_logs_l = sum(logs_l)                                  # SUM_LOGS_L
            m = len(self.n_components)                                # M
            
            # Iterate through al LOGS_L
            for log_l in logs_l:
                # FORMULA: DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
                other_words_likelihood = (sum_logs_l - log_l) / (m - 1)
                dic_scores.append(log_l - other_words_likelihood)
        except Exception as e:
            pass

        # Set the States to max DIC Score found else constant.
        states = self.n_components[np.argmax(dic_scores)] if dic_scores else self.n_constant
        
        return self.base_model(states)


#
# CLASS: SelectorCV
#
class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV
        # raise NotImplementedError
        
        mean_scores = []            # Mean_Scores list
        split_method = KFold()      # Save reference to 'KFold' in variable per shown in notebook
        
        # Use try/catch in case of exception
        try:
            # Iterate through all N_Components
            for n_component in self.n_components:
                model = self.base_model(n_component)   # Start w/the Base Model
                fold_scores = []                       # Fold and calculate model mean scores
                
                for _, test_idx in split_method.split(self.sequences):
                    test_X, test_length = combine_sequences(test_idx, self.sequences) # Get test sequences
                    fold_scores.append(model.score(test_X, test_length))              # Record each model score
                mean_scores.append(np.mean(fold_scores))                              # Compute mean of all fold scores
                
        except Exception as e:
            pass

        # Set the States to max CV Score found else constant.
        states = self.n_components[np.argmax(mean_scores)] if mean_scores else self.n_constant
        
        return self.base_model(states)
        
        
        
        
        