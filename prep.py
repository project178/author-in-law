from re import match, search, sub
from csv import DictReader
from codecs import open
from math import ceil
from re import sub
from collections import Counter
from os.path import exists


from joblib import dump, load
from allennlp.commands.elmo import ElmoEmbedder
from numpy import asarray
from rnnmorph.predictor import RNNMorphPredictor
from gensim import corpora
from gensim.models import Word2Vec, FastText
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from bert import bert_tokenization
from sklearn.feature_extraction.text import TfidfVectorizer
from glove import Corpus, Glove



def divise(text, chars=1000):

    def decide(x1, x2, chars, text, parts):
        
        if x1 >= 0:
            if x2 >= 0:
                if chars-x1 <= x2:
                    parts.append(text[:x1])
                    text = text[x1:]
                else:
                    parts.append(text[:chars+x2])
                    text = text[chars+x2:]
            else:
                parts.append(text[:x1])
                text = text[x1:]
        elif x2 >= 0:
            parts.append(text[:chars+x2])
            text = text[chars+x2:]
            
        else: return False
        
        if len(text) < 300:
            text = parts[-1] + text
            parts.pop()
            
        return text, parts
    
    parts = []

    while True:
        part1, part2 = text[:chars], text[chars:]
        n = decide(part1.rfind("/n"), part2.find("/n", 0, -10), chars, text, parts)
        print(part1, n, part1.rfind("."))
        text1, parts = decide(part1.rfind("."), part2.find(".", 0, -10), chars, text, parts) if n == False else n
        del part1, part2, n
        if len(text1) <= 500 or text == text1:
            parts.append(text1)
            del text, text1
            
            return parts
        
        else:
            text = text1
            del text1


def prepare(path_in="data/dataset1.csv", path_out="data/text"):
    
    authors_and_texts = {}
    with open(path_in, encoding="utf8") as inp:
        data = DictReader(inp, delimiter=";")
        for line in data:
            try: authors_and_texts[line["author"]].append(line["text"])
            except KeyError: authors_and_texts[line["author"]] = [line["text"]]

    for author in authors_and_texts.keys():
        for text in authors_and_texts[author]:
            l = len(text)
            if l <= 800: authors_and_texts[author].remove(text)
            if len(authors_and_texts[author]) == 1:
                if l <= 1600:
                    if len(authors_and_texts[author]) == 1:
                        authors_and_texts[author] += divise(text, 800)
                        authors_and_texts[author].remove(text)
                else:
                    authors_and_texts[author] += divise(text)
                    authors_and_texts[author].remove(text)
            del l
            
    texts = list(authors_and_texts.values())
    del authors_and_texts
    texts = [[part.replace("\'", "") for part in text] for text in texts]
    with open(path_out, "wb") as text_corp: dump(texts, text_corp)
    
    return texts

def get_stop_words(texts_path="data/text"):
    
    with open("data/text", "rb") as data: texts = load(data)
    c = Counter()
    data = [text for texts in texts.values() for text in texts]
    predictor = RNNMorphPredictor(language="ru")
    for line in data:
        forms = predictor.predict(sub("[^АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя\s]", "", line).lower().split())
        for form in forms: c[form.normal_form] += 1
    stopwords = set(w for w in dict(c).keys() if (100 > c[w] > 1000 or len(w) == 1))
    with open("data/stopwords", "wb") as tmp: dump(stopwords, tmp)
    
    return stopwords

def get_embeddings(dataset="data/text", embedding="tfidf"):
    
    with open(dataset, "rb") as data: texts = load(data)
    if not exists("data/stopwords"): get_stop_words(texts_path="data/text")
    predictor = RNNMorphPredictor(language="ru")
    with open("data/stopwords", "rb") as stopwords:
        new_texts = []
        for text in texts:
            new_text = []
            for part in text:
                new_part = []
                for form in predictor.predict(sub("[^АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя\s]", "", part).lower().split()): new_part += [] if (form.normal_form in stopwords or len(form.normal_form) == 1) else [form.normal_form]
                new_text.append(new_part[:-1])
            new_texts.append(new_text)
    texts = new_texts
    
    if embedding == "tfidf": return get_tfidf(texts)            
    elif embedding == "w2v": return get_w2v(texts, get_dict(texts))
    elif embedding == "d2v": return get_d2v(texts, get_dict(texts))
    elif embedding == "ft": return get_ft(texts)
    elif embedding == "glove": return get_glove(texts)
    elif embedding == "bert": return get_bert(texts)        

    
def get_tfidf(texts):
    
    vectorizer = TfidfVectorizer()
    vectorizer.fit([part for text in texts for part in text])
    tfidf_corp = [[vectorizer.transform([part]) for part in text] for text in texts]
    with open("data/tfidf", "wb") as embedding: dump(tfidf_corp, embedding)
    with open("data/tfidf_params", "wb") as params: dump(vectorizer.get_params(), params)

    return vectorizer


def get_dict(texts):

    dictionary = corpora.Dictionary(part for text in new_texts for part in text)
    with open("data/bow_dict", "wb") as tmp: dump(dictionary, tmp)

    return dictionary


def get_w2v(texts, dictionary):

    bow_corpus = [dictionary.doc2bow(part) for text in new_texts for part in text] 
    with open("data/w2v", "wb") as tmp: dump(bow_corpus, tmp)

    return bow_corpus


def get_d2v(texts, dictionary):

    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate([part for text in new_texts for part in text])]
    d2v = Doc2Vec(documents, vector_size=5, min_count=1, workers=8)
    d2v.save("data/d2v_model")
    d2v_corpora = [[d2v.infer_vector(part) for part in text] for text in new_texts]
    with open("data/d2v", "wb") as tmp: dump(d2v_corpora, tmp)

    return d2v


def get_ft(texts):

    ft = FastText(min_count=100)
    sentences = [part for text in new_texts for part in text]
    ft.build_vocab(sentences)
    ft.train(sentences=sentences, total_examples=len(sentences), epochs=100, workers=8)
    ft.save("data/ft_model")
    ft_corpora = []
    for text in texts:
      text_tmp = []
      for part in text:
        part_tmp = []
        for word in part.lower().split():
          try: part_tmp.append(ft[word])
          except KeyError: pass
        text_tmp.append(part_tmp)
      ft_corpora.append(text_tmp)
    with open("data/ft", "wb") as tmp: dump(ft_corpora, tmp)

    return ft_corpora


def get_glove(texts):

    corpus = Corpus()
    corpus.fit([part for text in new_texts for part in text], window=10)
    glove = Glove(no_components=100, learning_rate=0.05)
    glove.fit(corpus.matrix, epochs=100, no_threads=8)
    with open("data/glove_model", "wb") as tmp: dump(glove, tmp)
    glove.add_dictionary(corpus.dictionary)
    glove_corpora = [[glove.transform_paragraph(part) for part in text] for text in new_texts]
    with open("data/glove", "wb") as tmp: dump(glove_corpora, tmp)

    return glove_corpora


def get_bert(texts):

    new_texts = []
    embeddings = extract_embeddings("data/bert_", texts[0], output_layer_num=4, poolings=[POOL_NSP, POOL_MAX])
    for text in texts:
      embeddings = extract_embeddings("data/bert_", text, output_layer_num=4, poolings=[POOL_NSP, POOL_MAX])
      new_texts.append(embeddings)
    with open("data/bert", "wb") as tmp: dump(new_texts, tmp)
    
    return new_texts


def generate_data(dataset_name, dataset_size=10000, test=[]):

    with open("data/" + dataset_name, "rb") as dataset_file: data = load(dataset_file)
    if dataset_name == "tfidf": texts = [[[tuple(word) for word in part.todense().tolist()] for part in text] for text in data if len(text)>1]
    elif dataset_name == "ft": texts = [[numpy.sum(part, axis=0).tolist() for part in text] for text in data if len(text)>1]
    elif dataset_name == "w2v" or dataset_name == "d2v": texts = [[part.tolist() for part in text] for text in data if len(text)>1]
    else: texts = [text for text in data if len(text)>1]
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
