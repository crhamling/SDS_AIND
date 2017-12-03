import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # TODO implement the recognizer
    # return probabilities, guesses

    X_lengths = test_set.get_all_Xlengths().values()
    
    for X, lengths in X_lengths:
        #initialize the probability and guess
        logL = {}                          
        best_guess = None   
        #used to keep track of current highest score
        highest_score = float("-inf")                
        
        for word, model in models.items():
            try:
                #use the model to get a score for the word
                word_score = model.score(X, lengths) 
                logL[word] = word_score             
                #Keep track of highest score and best guess
                if word_score > highest_score:
                    highest_score = word_score
                    best_guess = word
            except:
                logL[word] = float("-inf")

        guesses.append(best_guess)
        probabilities.append(logL)

    return probabilities, guesses
