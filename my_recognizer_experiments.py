import numpy as np
import pandas as pd
import timeit
from asl_data import AslDb
from my_model_selectors import SelectorConstant, SelectorCV, SelectorBIC, SelectorDIC
from my_recognizer import recognize
from asl_utils import show_errors


asl = AslDb()

def run_experiment(asl):
    run_features = ['custom', 'custom-delta-norm-polar']
    selectors = {"Constant" : SelectorConstant, "CV": SelectorCV, "BIC" : SelectorBIC, "DIC" : SelectorDIC}
    #selectors = {"DIC" : SelectorDIC}
    features = load_features(asl)
    print("Feature, Selector, Training Time, Recognize Time, WER")
    for feature in run_features:
        for selector_name, selector in selectors.items():
            experiment(asl, feature, features[feature], selector_name,selector)

def experiment(asl, feature_name, feature, selector_name, selector):
    start = timeit.default_timer()
    models = train_all_words(asl, feature, selector)
    t_time = timeit.default_timer()-start
    test_set = asl.build_test(feature)
    probabilities, guesses = recognize(models, test_set)
    r_time = timeit.default_timer()-start
    show_results(feature_name, selector_name, t_time, r_time, guesses, test_set)

def show_results(feature_name, selector_name, training_time, recognize_time, guesses, test_set):
    WER = wer(guesses, test_set)
    print("\n {}, {}, {}, {}, {}".format(feature_name, selector_name, training_time, recognize_time, WER))

def wer(guesses, test_set):
    S = 0
    N = len(test_set.wordlist)
    num_test_words = len(test_set.wordlist)
    if len(guesses) != num_test_words:
        print("Size of guesses must equal number of test words ({})!".format(num_test_words))
    for word_id in range(num_test_words):
        if guesses[word_id] != test_set.wordlist[word_id]:
            S += 1
    return S/N
def train_all_words(asl, features, model_selector):
    training = asl.build_training(features)
    sequences = training.get_all_sequences()
    Xlengths = training.get_all_Xlengths()
    model_dict = {}
    for word in training.words:
        model = model_selector(sequences, Xlengths, word,
                        n_constant=3).select()
        model_dict[word]=model
    return model_dict

def load_features(asl):
    load_means(asl)
    load_std(asl)
    features = dict()
    features['ground'] = ground_feature(asl)
    features['norm'] = norm_feature(asl)
    features['polar'] = polar_feature(asl)
    features['delta'] = delta_feature(asl)
    features['custom'] = custom_feature(asl)
    features['custom-delta-norm-polar'] = custom_dnp_feature(asl)
    return features

def load_means(asl):
    # Define the mean of each coordinate by speaker
    df_means = asl.df.groupby('speaker').mean()
    asl.df['left-x-mean']= asl.df['speaker'].map(df_means['left-x'])
    asl.df['left-y-mean']= asl.df['speaker'].map(df_means['left-y'])
    asl.df['right-x-mean']= asl.df['speaker'].map(df_means['right-x'])
    asl.df['right-y-mean']= asl.df['speaker'].map(df_means['right-y'])

def load_std(asl):
    # Define the standard deviation of each coordinate by speaker
    df_std = asl.df.groupby('speaker').std()
    asl.df['left-x-std']= asl.df['speaker'].map(df_std['left-x'])
    asl.df['left-y-std']= asl.df['speaker'].map(df_std['left-y'])
    asl.df['right-x-std']= asl.df['speaker'].map(df_std['right-x'])
    asl.df['right-y-std']= asl.df['speaker'].map(df_std['right-y'])

def ground_feature(asl):
    # Ground Feature
    asl.df['grnd-ry'] = asl.df['right-y'] - asl.df['nose-y']
    asl.df['grnd-rx'] = asl.df['right-x'] - asl.df['nose-x']
    asl.df['grnd-ly'] = asl.df['left-y'] - asl.df['nose-y']
    asl.df['grnd-lx'] = asl.df['left-x'] - asl.df['nose-x']
    features_ground = ['grnd-rx','grnd-ry','grnd-lx','grnd-ly']
    return features_ground

def norm_feature(asl):
    # Normalize Cartesian Coordinates Feature
    asl.df['norm-rx'] = (asl.df['right-x'] - asl.df['right-x-mean']) / asl.df['right-x-std']
    asl.df['norm-ry'] = (asl.df['right-y'] - asl.df['right-y-mean']) / asl.df['right-y-std']
    asl.df['norm-lx'] = (asl.df['left-x'] - asl.df['left-x-mean']) / asl.df['left-x-std']
    asl.df['norm-ly'] = (asl.df['left-y'] - asl.df['left-y-mean']) / asl.df['left-y-std']
    features_norm = ['norm-rx', 'norm-ry', 'norm-lx','norm-ly']
    return features_norm

def polar_feature(asl):
    # Polar Coordinates Feature
    asl.df['polar-rr'] = np.hypot(asl.df['grnd-rx'], asl.df['grnd-ry'])
    asl.df['polar-rtheta'] = np.arctan2(np.array(asl.df['grnd-rx']), np.array(asl.df['grnd-ry']))
    asl.df['polar-lr'] = np.hypot(asl.df['grnd-lx'], asl.df['grnd-ly'])
    asl.df['polar-ltheta'] = np.arctan2(np.array(asl.df['grnd-lx']), np.array(asl.df['grnd-ly']))
    features_polar = ['polar-rr', 'polar-rtheta', 'polar-lr', 'polar-ltheta']
    return features_polar

def delta_feature(asl):
    # Delta Feature
    asl.df['delta-rx'] = asl.df['right-x'].diff().fillna(0)
    asl.df['delta-ry'] = asl.df['right-y'].diff().fillna(0)
    asl.df['delta-lx'] = asl.df['left-x'].diff().fillna(0)
    asl.df['delta-ly'] = asl.df['left-y'].diff().fillna(0)
    features_delta = ['delta-rx', 'delta-ry', 'delta-lx', 'delta-ly']
    return features_delta

def custom_dnp_feature(asl):
    df_means = asl.df.groupby('speaker').mean()
    df_std = asl.df.groupby('speaker').std()
    asl.df['polar-rr-mean']= asl.df['speaker'].map(df_means['polar-rr'])
    asl.df['polar-rtheta-mean']= asl.df['speaker'].map(df_means['polar-rtheta'])
    asl.df['polar-lr-mean']= asl.df['speaker'].map(df_means['polar-lr'])
    asl.df['polar-ltheta-mean']= asl.df['speaker'].map(df_means['polar-ltheta'])

    asl.df['polar-rr-std']= asl.df['speaker'].map(df_std['polar-rr'])
    asl.df['polar-rtheta-std']= asl.df['speaker'].map(df_std['polar-rtheta'])
    asl.df['polar-lr-std']= asl.df['speaker'].map(df_std['polar-lr'])
    asl.df['polar-ltheta-std']= asl.df['speaker'].map(df_std['polar-ltheta'])

    asl.df['np-rr'] = (asl.df['polar-rr'] - asl.df['polar-rr-mean']) / asl.df['polar-rr-std']
    asl.df['np-rtheta'] = (asl.df['polar-rtheta'] - asl.df['polar-rtheta-mean']) / asl.df['polar-rtheta-std']
    asl.df['np-lr'] = (asl.df['polar-lr'] - asl.df['polar-lr-mean']) / asl.df['polar-lr-std']
    asl.df['np-ltheta'] = (asl.df['polar-ltheta'] - asl.df['polar-ltheta-mean']) / asl.df['polar-ltheta-std']

    asl.df['dnp-rr'] = asl.df['np-rr'].diff().fillna(0)
    asl.df['dnp-rtheta'] = asl.df['np-rtheta'].diff().fillna(0)
    asl.df['dnp-lr'] = asl.df['np-lr'].diff().fillna(0)
    asl.df['dnp-ltheta'] = asl.df['np-ltheta'].diff().fillna(0)

    features_custom = ['dnp-rr', 'dnp-rtheta', 'dnp-lr', 'dnp-ltheta']
    return features_custom

def custom_feature(asl):
    # Custom Feature
    df_means = asl.df.groupby('speaker').mean()
    df_std = asl.df.groupby('speaker').std()

    asl.df['grnd-rx-mean']= asl.df['speaker'].map(df_means['grnd-rx'])
    asl.df['grnd-ry-mean']= asl.df['speaker'].map(df_means['grnd-ry'])
    asl.df['grnd-lx-mean']= asl.df['speaker'].map(df_means['grnd-lx'])
    asl.df['grnd-ly-mean']= asl.df['speaker'].map(df_means['grnd-ly'])

    asl.df['grnd-rx-std']= asl.df['speaker'].map(df_std['grnd-rx'])
    asl.df['grnd-ry-std']= asl.df['speaker'].map(df_std['grnd-ry'])
    asl.df['grnd-lx-std']= asl.df['speaker'].map(df_std['grnd-lx'])
    asl.df['grnd-ly-std']= asl.df['speaker'].map(df_std['grnd-ly'])

    asl.df['ng-rx'] = (asl.df['grnd-rx'] - asl.df['grnd-rx-mean']) / asl.df['grnd-rx-std']
    asl.df['ng-ry'] = (asl.df['grnd-ry'] - asl.df['grnd-ry-mean']) / asl.df['grnd-ry-std']
    asl.df['ng-lx'] = (asl.df['grnd-lx'] - asl.df['grnd-lx-mean']) / asl.df['grnd-lx-std']
    asl.df['ng-ly'] = (asl.df['grnd-ly'] - asl.df['grnd-ly-mean']) / asl.df['grnd-ly-std']
    features_custom = ['ng-rx', 'ng-ry', 'ng-lx', 'ng-ly']

    return features_custom

run_experiment(asl)
