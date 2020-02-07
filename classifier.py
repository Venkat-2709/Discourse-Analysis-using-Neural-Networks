import numpy as np

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


def head_classification(X, y):
    label_encoder = LabelEncoder()
    encoded_annotation = label_encoder.fit_transform(y)
    onehot_encoder = OneHotEncoder()
    encoded_annotation = np.reshape(encoded_annotation, (-1, 1))
    Y = onehot_encoder.fit_transform(encoded_annotation).toarray()
    Y = Y[:, 1:]

    classifier = RandomForestClassifier(n_estimators=1000, criterion='entropy', random_state=0)
    classifier.fit(X, Y)

    return classifier


def body_classification(X_1, y_1):
    label_encoder = LabelEncoder()
    onehot_encoder = OneHotEncoder()
    encoded_annotation_1 = label_encoder.fit_transform(y_1)
    encoded_annotation_1 = np.reshape(encoded_annotation_1, (-1, 1))
    Y_1 = onehot_encoder.fit_transform(encoded_annotation_1).toarray()
    Y_1 = Y_1[:, 1:]

    model = Sequential()
    model.add(Dense(300, activation='relu', input_dim=600))
    model.add(Dense(150, activation='relu'))
    model.add(Dense(8, activation='sigmoid'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(X_1, Y_1, batch_size=10, epochs=2)

    return model