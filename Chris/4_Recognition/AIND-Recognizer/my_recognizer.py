import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probs = []
    guesses = []

    X_lengths = test_set.get_all_Xlengths()
    
    for X, lengths in X_lengths.values():
        log_l = {}
        best_score = float("-inf")
        best = None 
        
        for word, model in models.items():
            try:
                word_score = model.score(X, lengths)
                log_l[word] = word_score
                
                if word_score > best_score:
                    best_score = word_score
                    best = word
            except:
                log_l[word] = float("-inf")

        guesses.append(best)
        probs.append(log_l)

    return probs, guesses