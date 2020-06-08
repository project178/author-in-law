import download
import train
import prep

if __name__=="main":
    embeddings_names = ["tfidf", "w2v", "d2v", "ft", "glove", "bert"]
    models_names = ["regrression", "dense", "cnn", "lstm", "siamese"]
    download.relevant_text_from_site_to_csv(site="http://istmat.info/documents?tid_theme=All&tid_area=All&tid_state=All&tid_type=All&tid_tags=&period=&title=%D0%B4%D0%BE%D0%BF%D1%80%D0%BE%D1%81&page=", dataset="dataset1.csv")
    prep.prepare()
    for embedding in embeddings_names:
        prep.get_embeddings(embedding=embedding)
        for model in models_names:
            print("\n" + embedding + " + " + model + " score: ")
            train.train(embedding, model)
