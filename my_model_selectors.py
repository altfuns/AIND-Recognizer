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
        best_num_components = None
        min_bic_score = float("inf")
        for n in range(self.min_n_components, self.max_n_components + 1):
            try:
                new_model = self.base_model(n)
                X, lengths = self.hwords[self.this_word]
                logL = new_model.score(X, lengths)
                #p = n^2 + 2*d*n - 1
                #  BIC = −2 log L + p log N
                p = n**2 + 2 * len(X[1]) * n - 1
                bic_score = -2 * logL + p * math.log(len(X))
                if bic_score < min_bic_score:
                    min_bic_score = bic_score
                    best_num_components = n
            except:
                if self.verbose:
                    print("failure on {} with {} states".format(self.this_word, n))
        return self.base_model(best_num_components)


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        """ select best model based on Discriminative Information Criterion (DIC)
            DIC is distante between likelihood of the word and the average likelihood of the other words
        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # M number of categories or classe (words)
        M = len(self.words)
        best_num_components = self.n_constant #sets the default value to return when all the models fail
        max_dic_score = float("-inf")
        for n in range(self.min_n_components, self.max_n_components + 1):
            try:
                new_model = self.base_model(n)
                X, lengths = self.hwords[self.this_word]
                # Compute the P(X(i)) formula part. The Likelihood of the word
                logL = new_model.score(X, lengths)
                antiL = 0
                for word in self.words:
                    if self.this_word != word:
                        X, lengths = self.hwords[word]
                        try:
                            # SUM the Likelihoods of the other words
                            antiL = antiL + new_model.score(X, lengths)
                        except:
                            if self.verbose:
                                print("failure score on {} with {} states".format(word, n))
                # Compute the diference between the word Likelihood and the average Likelihood of the other words
                dic_score = logL - (1/(M-1) * antiL)
                if dic_score > max_dic_score:
                    max_dic_score = dic_score
                    best_num_components = n
            except:
                if self.verbose:
                    print("failure on {} with {} states".format(self.this_word, n))
        return self.base_model(best_num_components)


class SelectorCV(ModelSelector):
    """ select best model based on average log Likelihood of cross-validation folds
    :return: GaussianHMM object
    """

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # Return a model with the default number of component
        # when the word hasn't enough number of samples
        if min(len(self.sequences), 3) < 3:
            return self.base_model(self.n_constant)
        split_method = KFold()
        best_num_components = None
        best_avg_logL = float("-inf")
        for num_components in range(self.min_n_components, self.max_n_components + 1):
            try:
                logLs = []
                for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                    # Set the mode with the train Xs
                    self.X, self.lengths = combine_sequences(cv_train_idx, self.sequences)
                    new_model = self.base_model(num_components)
                    X, lengths = combine_sequences(cv_test_idx, self.sequences)
                    # Compute the Likelihood evaluating the test samples
                    logLs.append(new_model.score(X, lengths))
                # Compute the mean of the log Likelihood
                avg_logL = np.mean(logLs)
                # Maximize the average Likelihood for the num_components
                if avg_logL > best_avg_logL:
                    best_avg_logL = avg_logL
                    best_num_components = num_components
            except:
                if self.verbose:
                    print("failure on {} with {} states".format(self.this_word, num_components))
        # Restore the X,lengths inital values
        self.X, self.lengths = self.hwords[self.this_word]
        return self.base_model(best_num_components)
