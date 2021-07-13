import librosa
import pandas as pd
import numpy as np
import glob

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

import librosa.display
import sklearn
from sklearn import preprocessing

classical_filepath = glob.glob("./Music-data/classical/*.au")
rock_filepath = glob.glob("./Music-data/rock/*.au")

classical = [librosa.load(track)[0] for track in classical_filepath]
rock = [librosa.load(track)[0] for track in rock_filepath]


def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)


def extract_features(signal):
    return [
        np.mean(librosa.feature.zero_crossing_rate(signal)),
        np.mean(librosa.feature.spectral_centroid(signal)),
        np.mean(librosa.feature.mfcc(signal)),
        np.mean(librosa.feature.rms(signal, center=True)),
        np.mean(librosa.feature.chroma_stft(signal))
    ]


classical_features = np.array([extract_features(x) for x in classical])
rock_features = np.array([extract_features(x) for x in rock])

feature_classical = pd.DataFrame(classical_features, columns=['zero crossing', 'spectral centroid',
                                                              'mfcc', 'rmse', 'chroma'])
feature_classical['category'] = 'classical'
feature_rock = pd.DataFrame(rock_features, columns=['zero crossing', 'spectral centroid',
                                                    'mfcc', 'rmse', 'chroma'])
feature_rock['category'] = 'rock'
final_data = pd.concat([feature_classical, feature_rock]).reset_index()

final_data['category'] = final_data['category'] == 'classical'

X = final_data[['zero crossing', 'spectral centroid', 'mfcc', 'rmse', 'chroma']]
Y = [int(c) for c in final_data['category']]
X_scaled = preprocessing.scale(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y, shuffle=True, random_state=2,
                                                    test_size=0.2)

logistic_regressor = LogisticRegression()
logistic_regressor.fit(X_train, y_train)

print(logistic_regressor.score(X_test, y_test))
