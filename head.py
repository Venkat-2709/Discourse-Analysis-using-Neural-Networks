import numpy as np


def head_comments(sub_annotations, all_bodies, subreddit, model):
    first_body = []
    first_annotation = []
    for i in range(len(all_bodies)):
        body_1 = all_bodies[i][0]
        annotation_1 = sub_annotations[i][0]
        if body_1 == '':
            body_1 = subreddit[i]
        if annotation_1 != None:
            if body_1 != None:
                first_body.append(body_1)
                first_annotation.append(annotation_1)

    vector_list = []
    for sentence in first_body:
        splt_sent = sentence.split()
        count_words = 0
        vectors = np.zeros(300)
        for word in splt_sent:
            count_words += 1
            if word in model:
                vectors = np.add(vectors, model[word])
            else:
                vectors = np.add(vectors, np.zeros(300))

        vector_s = np.divide(vectors, count_words)
        vector_list.append(vector_s)

    X = np.asarray(vector_list)
    y = np.asarray(first_annotation)
    y = y.reshape((-1, 1))

    return X, y