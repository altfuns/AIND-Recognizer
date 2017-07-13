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
    for i in range(0, test_set.num_items):
        X, lengths = test_set.get_item_Xlengths(i)
        word_probs = dict()
        guess_word = None
        guess_prob = float("-inf")
        for word, model in models.items():
            try:
                logL = model.score(X, lengths)
                word_probs.update({word : logL})
                if logL > guess_prob:
                    guess_word = word
                    guess_prob = logL
            except:
                pass
                #print("failure computing score for word:", word)
        probabilities.append(word_probs)
        guesses.append(guess_word)
    return (probabilities, guesses)
