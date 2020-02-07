import json
import gensim.models.keyedvectors as word2vec

from head_cleaner import set_data, clean_data
from head import head_comments
from classifier import head_classification, body_classification
from body_cleaner import body_data

from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer

if __name__ == "__main__":
    with open("coarse_discourse_dump_reddit.jsonlist", "r") as R:
        data = [json.loads(l) for l in R.readlines()]
    print('Data Imported.')

    model = word2vec.KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)
    print('Word2vec Loaded.')

    subreddit, posts = set_data(data)
    tokenizer = RegexpTokenizer(r'\w+')
    lemmatizer = WordNetLemmatizer()

    sub_annotations, all_bodies, subreddit, posts = clean_data(tokenizer, lemmatizer, subreddit, posts)
    print('Data Cleaned and Preprocessed.')

    X_train, Y_train = head_comments(sub_annotations, all_bodies, subreddit, model)

    head_model = head_classification(X_train, Y_train)
    print('First Model Trained.')

    X_train_b, Y_train_b = body_data(model, posts)

    body_model = body_classification(X_train_b, Y_train_b)
    print('Body Model Trained.')