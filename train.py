from sklearn.linear_model import LogisticRegression
from codecs import open
from joblib import dump, load
import numpy
from random import choice
from keras.models import Sequential, Model
from keras.layers import Dense, Conv1D, MaxPooling1D, Conv2D, MaxPooling2D, Dropout, Flatten, LSTM, Lambda, Input
from keras import backend as K
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
from keras_bert import extract_embeddings, POOL_NSP, POOL_MAX
from tqdm import tqdm

with open("data/text", "rb") as dataset_file: data = load(dataset_file)
model_path = 'data/bert_'
texts = []
embeddings = extract_embeddings(model_path, data[0], output_layer_num=4, poolings=[POOL_NSP, POOL_MAX])
print(numpy.asarray(embeddings).shape)
for text in data:
  embeddings = extract_embeddings(model_path, text, output_layer_num=4, poolings=[POOL_NSP, POOL_MAX])
  texts.append(embeddings)
print(numpy.asarray(texts).shape)

with open("data/text", "rb") as dataset_file: data = load(dataset_file)
model_path = 'data/bert_'
for i in tqdm(range(186, len(data))):
  with open("data/trash/"+str(i), "wb") as tmp:
    dump(extract_embeddings(model_path, data[i], output_layer_num=4, poolings=[POOL_NSP, POOL_MAX]), tmp)

text = []
for i in tqdm(range(195)):
  with open("data/trash/"+str(i), "rb") as tmp: text.append(load(tmp))
with open("data/bert_new", "wb") as tmp: dump(text, tmp)

def generate_data(dataset_name, dataset_size=10000, test=[]):

  with open("data/" + dataset_name, "rb") as dataset_file: data = load(dataset_file)
  texts = [text for text in data if len(text)>1]
  del data
  if "bert" in dataset_name:
    l = 0
    for text in texts:
      for part in text: l = max(l, len(part))
    for text in texts:
      for part in text:
        l1 = len(part)
        while len(part) < l: part.append(0)
        part.append(l1)
  XY = set()
  while len(XY) < dataset_size/2:
    same1 = same2 = 0
    while [same1, same2] in test or [same2, same1] in test or same2 == same1:
      same = choice(texts)
      same1 = tuple(choice(same))
      same2 = tuple(choice(same))
    if len({same1, same2}) == 2: XY.add((frozenset({same1, same2}), 1))
    del same, same1, same2
  while len(XY) < dataset_size:
    difpart2 = difpart1 = 0
    while difpart2 == difpart1 or [dif1, dif2] in test or [dif2, dif1] in test:
      difpart1 = choice(texts)
      difpart2 = choice(texts)
      dif1, dif2 = choice(difpart1), choice(difpart2)
    XY.add((frozenset({tuple(dif1), tuple(dif2)}), 0))
    del difpart1, difpart2, dif1, dif2
  X = []
  Y = []
  for xy in XY:
    X.append([list(text) for text in xy[0]])
    Y.append(xy[1])  
    
  return X, Y

def train(dataset_name, model_name):

  test_X, test_Y = generate_data(dataset_name, dataset_size=1000)
  X, Y = generate_data(dataset_name, test=test_X)
  X, Y, test_X, test_Y = numpy.array(X), numpy.array(Y), numpy.array(test_X), numpy.array(test_Y)
  nsamples, nx, ny = X.shape
  X = X.reshape((nsamples,nx*ny))
  nsamples, nx, ny = test_X.shape
  test_X = test_X.reshape((nsamples,nx*ny))

  if model_name=="regression":
    nsamples, nx, ny = X.shape
    X = X.reshape((nsamples,nx*ny))
    reg = LogisticRegression().fit(X, Y)
    del X, Y, nsamples, nx, ny
    with open("data/models/" + model_name + dataset_name, "wb") as tmp: dump(reg.get_params, tmp)
    nsamples, nx, ny = test_X.shape
    test_X = test_X.reshape((nsamples,nx*ny))
    print(reg.score(test_X, test_Y))
    return reg
  
  elif model_name=="dense":
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

  elif model_name=="cnn":
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
  
  elif model_name=="lstm":
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
  
  elif model_name=="siamese":

X = X.tolist()
Y = Y.tolist()
for i, x in enumerate(X):
  if len(x) != 2:
    print(x)
    X.remove(x)
    Y.pop(i)
X, Y = numpy.asarray(X), numpy.asarray(Y)

test_X = test_X.tolist()
test_Y = test_Y.tolist()
for i, x in enumerate(test_X):
  if len(x) != 2:
    print(x)
    test_X.remove(x)
    test_Y.pop(i)
test_X, test_Y = numpy.asarray(test_X), numpy.asarray(test_Y)

def generate_data_d2v(dataset_name, dataset_size=10000, test=[]):

  with open("data/" + dataset_name, "rb") as dataset_file: data = load(dataset_file)
  texts = [[part.tolist() for part in text] for text in data if len(text)>1]
  del data
  if "bert" in dataset_name:
    l = 0
    for text in texts:
      for part in text: l = max(l, len(part))
    for text in texts:
      for part in text:
        l1 = len(part)
        while len(part) < l: part.append(0)
        part.append(l1)
  XY = set()
  while len(XY) < dataset_size/2:
    same1 = same2 = 0
    while [same1, same2] in test or [same2, same1] in test or same2 == same1:
      same = choice(texts)
      same1 = tuple(choice(same))
      same2 = tuple(choice(same))
    if len({same1, same2}) == 2: XY.add((frozenset({same1, same2}), 1))
    del same, same1, same2
  while len(XY) < dataset_size:
    difpart2 = difpart1 = 0
    while difpart2 is difpart1 or [dif1, dif2] in test or [dif2, dif1] in test:
      difpart1 = choice(texts)
      difpart2 = choice(texts)
      dif1, dif2 = choice(difpart1), choice(difpart2)
    XY.add((frozenset({tuple(dif1), tuple(dif2)}), 0))
    del difpart1, difpart2, dif1, dif2
  X = []
  Y = []
  for xy in XY:
    X.append([list(text) for text in xy[0]])
    Y.append(xy[1])  
    
  return X, Y

def generate_data_ft(dataset_name, dataset_size=10000, test=[]):

  with open("data/" + dataset_name, "rb") as dataset_file: data = load(dataset_file)
  texts = [[numpy.sum(part, axis=0).tolist() for part in text] for text in data if len(text)>1]
  del data
  if "bert" in dataset_name:
    l = 0
    for text in texts:
      for part in text: l = max(l, len(part))
    for text in texts:
      for part in text:
        l1 = len(part)
        while len(part) < l: part.append(0)
        part.append(l1)
  XY = set()
  while len(XY) < dataset_size/2:
    same1 = same2 = 0
    while [same1, same2] in test or [same2, same1] in test or same2 is same1:
      same = choice(texts)
      same1 = tuple(choice(same))
      same2 = tuple(choice(same))
    if len({same1, same2}) == 2: XY.add((frozenset({same1, same2}), 1))
    del same, same1, same2
  while len(XY) < dataset_size:
    difpart2 = difpart1 = 0
    try:
      while difpart2 is difpart1 or [dif1, dif2] in test or [dif2, dif1] in test:
        difpart1 = choice(texts)
        difpart2 = choice(texts)
        dif1, dif2 = choice(difpart1), choice(difpart2)
    except ValueError: print(type(difpart1), type(difpart2), type(dif1), type(dif2), type(test))
    XY.add((frozenset({tuple(dif1), tuple(dif2)}), 0))
    del difpart1, difpart2, dif1, dif2
  X = []
  Y = []
  for xy in XY:
    X.append([list(text) for text in xy[0]])
    Y.append(xy[1])  
    
  return X, Y

def generate_data_tfidf(dataset_name, dataset_size=10000, test=[]):

  with open("data/" + dataset_name, "rb") as dataset_file: data = load(dataset_file)
  texts = [[[tuple(word) for word in part.todense().tolist()] for part in text] for text in data if len(text)>1]
  del data
  if "bert" in dataset_name:
    l = 0
    for text in texts:
      for part in text: l = max(l, len(part))
    for text in texts:
      for part in text:
        l1 = len(part)
        while len(part) < l: part.append(0)
        part.append(l1)
  XY = set()
  while len(XY) < dataset_size/2:
    same1 = same2 = 0
    while [same1, same2] in test or [same2, same1] in test or same2 is same1:
      same = choice(texts)
      same1 = tuple(choice(same))
      same2 = tuple(choice(same))
    if len({same1, same2}) == 2: XY.add((frozenset({same1, same2}), 1))
    del same, same1, same2
  while len(XY) < dataset_size:
    difpart2 = difpart1 = 0
    while difpart2 is difpart1 or [dif1, dif2] in test or [dif2, dif1] in test:
      difpart1 = choice(texts)
      difpart2 = choice(texts)
      dif1, dif2 = choice(difpart1), choice(difpart2)
    XY.add((frozenset({tuple(dif1), tuple(dif2)}), 0))
    del difpart1, difpart2, dif1, dif2
  X = []
  Y = []
  for xy in XY:
    X.append([list(text) for text in xy[0]])
    Y.append(xy[1])  
    
  return X, Y

print(cosine_similarity(test_X[0]))
pred = [1 if cosine_similarity(x)[0][0] == 1. else 0 for x in test_X]
print(numpy.array(pred))
print(test_Y)
score = compute_accuracy(test_Y, numpy.array(pred))
print(score)

test_X, test_Y = generate_data_d2v("bert_new", dataset_size=1000)
X, Y = generate_data_d2v("bert_new", test=test_X, dataset_size=10000)
X, Y, test_X, test_Y = numpy.asarray([numpy.asarray([numpy.asarray(x) for x in pair]) for pair in X]), numpy.asarray(Y), numpy.asarray([numpy.asarray([numpy.asarray(x) for x in pair]) for pair in test_X]), numpy.asarray(test_Y)
nsamples, nx, ny = X.shape
X1 = X.reshape((nsamples,nx*ny))
nsamples, nx, ny = test_X.shape
test_X1 = test_X.reshape((nsamples,nx*ny))

for x in X:
  if x.shape != (2, 100): print(x.shape)

reg = LogisticRegression().fit(X1, Y)
with open("data/regression_bert", "wb") as tmp: dump(reg.get_params, tmp)
print(reg.score(test_X1, test_Y))

model = Sequential()
model.add(Dense(256, activation="relu"))
model.add(Dense(128, activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(16, activation="relu"))
model.add(Dense(4, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X1, Y, epochs=1)
with open("data/dense_bert", "wb") as tmp: dump(model, tmp)
score = model.evaluate(test_X1, test_Y)
print(score)

model = Sequential()
model.add(Conv1D(64, kernel_size=1, activation='relu'))
model.add(MaxPooling1D(pool_size=(1)))
model.add(Conv1D(32, kernel_size=1, activation='relu'))
model.add(MaxPooling1D(pool_size=(1)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation="sigmoid"))
model.compile(optimizer='Adagrad', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, Y, epochs=14)
with open("data/cnn_bert", "wb") as tmp: dump(model, tmp)
score = model.evaluate(test_X, test_Y)
print(score)

test_X1 = numpy.expand_dims(test_X1, axis=1)
print(test_X1.shape)

model = Sequential()
model.add(LSTM(16, input_shape=(2, 6145)))
model.add(Dense(1))
model.compile(optimizer='Adagrad', loss='binary_crossentropy', metrics=['binary_accuracy'])
model.fit(X, Y, epochs=4)
with open("data/lstm_bert", "wb") as tmp: dump(model, tmp)
score = model.evaluate(test_X, test_Y)
print(score)

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
    return numpy.mean(pred == y_true)

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

input_a = Input(shape=(1, 6145))
input_b = Input(shape=(1, 6145))
processed_a = base(input_a)
processed_b = base(input_b)
distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])
model = Model([input_a, input_b], distance)
model.compile(optimizer='Adagrad', loss=contrastive_loss, metrics=[accuracy])
model.fit([numpy.expand_dims(X[:, 0], axis=1), numpy.expand_dims(X[:, 1], axis=1)], Y, epochs=23)
with open("data/siamese_glove", "wb") as tmp: dump(model, tmp)
y_pred = model.predict([numpy.expand_dims(test_X[:, 0], axis=1), numpy.expand_dims(test_X[:, 1], axis=1)])
score = compute_accuracy(test_Y, y_pred)
print(score)

def generate_data_tfidf(dataset_name, dataset_size=10000, test=[]):

  with open("data/" + dataset_name, "rb") as dataset_file: data = load(dataset_file)
  texts = [[part.tolil() for part in text] for text in data if len(text)>1]
  del data
  for text in texts:
    for part in text:
      for word in part: print(word[0])
  XY = set()
  while len(XY) < dataset_size/2:
    same1 = same2 = 0
    while [same1, same2] in test or [same2, same1] in test or same2 == same1:
      same = choice(texts)
      same1 = tuple(choice(same))
      same2 = tuple(choice(same))
    XY.add((frozenset({same1, same2}), 1))
    del same, same1, same2
  while len(XY) < dataset_size:
    difpart2 = difpart1 = 0
    while difpart2 == difpart1 or [dif1, dif2] in test or [dif2, dif1] in test:
      difpart1 = choice(texts)
      difpart2 = choice(texts)
      dif1, dif2 = choice(difpart1), choice(difpart2)
    XY.add((frozenset({tuple(dif1), tuple(dif2)}), 0))
    del difpart1, difpart2, dif1, dif2
  X = []
  Y = []
  for xy in XY:
    X.append([list(text) for text in xy[0]])
    Y.append(xy[1])  
    
  return X, Y

def test(test_data, model_name, dataset_name):

  if model_name=="regression":
    reg = LogisticRegression()
    with open("data/models/" + model_name + dataset_name, "wb") as tmp: reg.set_params(load(tmp))
    nsamples, nx, ny, nn, nm = test_X.shape
    test_X = test_X.reshape((nsamples,nx*ny*nn*nm))
    return reg.score(test_X, test_Y)

with open("data/large_elmo", "rb") as dataset_file: data = load(dataset_file)
texts = [[numpy.sum(part, axis=0).tolist() for part in text] for text in data if len(text)>1]
texts = [text for text in data if len(text)>1]
del data

print(len(texts[-8][1][1]))
print(len(texts[0][0]))

XY = set()
test = []
while len(XY) < 10/2:
  same1 = same2 = 0
  while [same1, same2] in test or [same2, same1] in test or same2 == same1:
    same = choice(texts)
    same1 = tuple(choice(same))
    same2 = tuple(choice(same))
  XY.add((frozenset({same1, same2}), 1))
  del same, same1, same2
while len(XY) < 10:
  difpart2 = difpart1 = 0
  while difpart2 == difpart1 or [dif1, dif2] in test or [dif2, dif1] in test:
    difpart1 = choice(texts)
    difpart2 = choice(texts)
    dif1, dif2 = choice(difpart1), choice(difpart2)
  XY.add((frozenset({tuple(dif1), tuple(dif2)}), 0))
  del difpart1, difpart2, dif1, dif2
X = []
Y = []
for xy in XY:
  X.append([list(text) for text in xy[0]])
  Y.append(xy[1])
