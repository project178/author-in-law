from codecs import open

from joblib import dump, load
from numpy import array, expand_dims, mean
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
from keras.models import Sequential, Model
from keras.layers import Dense, Conv1D, MaxPooling1D, Conv2D, MaxPooling2D, Dropout, Flatten, LSTM, Lambda, Input
from keras import backend as K

import prep


def train(dataset_name, model_name):

    test_X, test_Y = prep.generate_data(dataset_name, dataset_size=1000)
    X, Y = prep.generate_data(dataset_name, test=test_X)
    X, Y, test_X, test_Y = array(X), array(Y), array(test_X), array(test_Y)
    nsamples, nx, ny = X.shape
    X = X.reshape((nsamples,nx*ny))
    nsamples, nx, ny = test_X.shape
    test_X = test_X.reshape((nsamples,nx*ny))
    
    if model_name=="regression": return fit_regression(X, Y, test_X, test_Y
                                                       
    elif model_name=="dense": return fit_dense(X, Y, test_X, test_Y)
    
    elif model_name=="cnn": return fit_cnn(X, Y, test_X, test_Y)
    
    elif model_name=="lstm": return fit_lstm(X, Y, test_X, test_Y)
    
    elif model_name=="siamese": return fit_siamese(X, Y, test_X, test_Y)
    


def fit_regression(X, Y, test_X, test_Y):

  nsamples, nx, ny = X.shape
  X = X.reshape((nsamples,nx*ny))
  reg = LogisticRegression().fit(X, Y)
  del X, Y, nsamples, nx, ny
  with open("data/models/" + model_name + dataset_name, "wb") as tmp: dump(reg.get_params, tmp)
  nsamples, nx, ny = test_X.shape
  test_X = test_X.reshape((nsamples,nx*ny))
  print(reg.score(test_X, test_Y))
  
  return reg


def fit_dense(X, Y, test_X, test_Y):

  model = Sequential()
  model.add(Dense(256, activation="relu"))
  model.add(Dense(128, activation="relu"))
  model.add(Dense(64, activation="relu"))
  model.add(Dense(16, activation="relu"))
  model.add(Dense(4, activation="relu"))
  model.add(Dense(1, activation="sigmoid"))
  model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
  model.fit(X, Y, epochs=10)
  del X, Y
  with open("data/" + model_name + dataset_name, "wb") as tmp: dump(model, tmp)
  score = model.evaluate(test_X, test_Y)
  print(score)
  
  return model


def fit_cnn(X, Y, test_X, test_Y):

  model = Sequential()
  model.add(Conv1D(64, kernel_size=1, activation='relu', input_shape=(2, 10384)))
  model.add(MaxPooling1D(pool_size=(1)))
  model.add(Conv1D(32, kernel_size=1, activation='relu'))
  model.add(MaxPooling1D(pool_size=(1)))
  model.add(Flatten())
  model.add(Dense(256, activation='relu'))
  model.add(Dense(1, activation="sigmoid"))
  model.compile(optimizer='Adagrad', loss='binary_crossentropy', metrics=['accuracy'])
  model.fit(X, Y, epochs=3)
  del X, Y
  with open("data/" + model_name + dataset_name, "wb") as tmp: dump(model, tmp)
  score = model.evaluate(test_X, test_Y)
  print(score)
  
  return model


def fit_lstm(X, Y, test_X, test_Y):

  model = Sequential()
  model.add(LSTM(1024, input_shape=(2, 10384)))
  model.add(Dense(1))
  model.compile(optimizer='Adagrad', loss='binary_crossentropy', metrics=['accuracy'])
  model.fit(X, Y, epochs=3)
  del X, Y
  with open("data/" + model_name + dataset_name, "wb") as tmp: dump(model, tmp)
  score = model.evaluate(test_X, test_Y)
  print(score)
  
  return model


def fit_siamese(X, Y, test_X, test_Y):

  input_a = Input(shape=(1, 6145))
  input_b = Input(shape=(1, 6145))
  processed_a = base(input_a)
  processed_b = base(input_b)
  distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])
  model = Model([input_a, input_b], distance)
  model.compile(optimizer='Adagrad', loss=contrastive_loss, metrics=[accuracy])
  model.fit([expand_dims(X[:, 0], axis=1), expand_dims(X[:, 1], axis=1)], Y, epochs=23)
  with open("data/siamese_glove", "wb") as tmp: dump(model, tmp)
  y_pred = model.predict([expand_dims(test_X[:, 0], axis=1), expand_dims(test_X[:, 1], axis=1)])
  score = compute_accuracy(test_Y, y_pred)
  print(score)
  
  return model


def euclidean_distance(vects):
  
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
  
    shape1, shape2 = shapes
    
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):

    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)


def accuracy(y_true, y_pred): return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))


def compute_accuracy(y_true, y_pred):

    pred = y_pred.ravel() < 0.5
    
    return mean(pred == y_true)


def base(inp):
  
  x = Conv1D(128, kernel_size=1, activation='relu')(inp)
  x = MaxPooling1D(pool_size=(1))(x)
  x = Conv1D(64, kernel_size=1, activation='relu')(x)
  x = MaxPooling1D(pool_size=(1))(x)
  x = Conv1D(32, kernel_size=1, activation='relu')(x)
  x = MaxPooling1D(pool_size=(1))(x)
  x = Conv1D(16, kernel_size=1, activation='relu')(x)
  x = MaxPooling1D(pool_size=(1))(x)
  x = Flatten()(x)
  x = Dense(16, activation='relu')(x)
  
  return x
