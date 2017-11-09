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
    # Implement the recognizer
    # Return probabilities, guesses

    # For each x and lengths in the test set
    for x, lengths in test_set.get_all_Xlengths().values():
        # Store the log of the likelihood of a word
        log_liklihood = {} 
        # Store the maximum score
        max_score = float("-inf")
        # Store the best score
        best_word = None 
        # For each word and model in the models
        for word, model in models.items():
            try: # Eliminate non-viable models from consideration
                # Store log of the liklihood score for the word on the fitted model
                log_liklihood[word] = model.score(x, lengths)  
                # If the word score is higher than the previous max score
                if log_liklihood[word] > max_score:
                    # Update the max score
                    max_score = log_liklihood[word]
                    # Save the word that produced the max score
                    best_word = word
            except:
                # If non-viable model, throw out word score
                log_liklihood[word] = float("-inf")

        # Add the best word to the list
        guesses.append(best_word)
        # Add the log of the liklihood to the list
        probabilities.append(log_liklihood)

    return probabilities, guesses