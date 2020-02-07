import numpy as np


def body_data(model, posts):
    for post_1 in posts:
        for each_post in post_1:
            body_1 = each_post.get('body')
            if body_1 != None:
                splt_body = body_1.split()
                count_words_1 = 0
                vectors_1 = np.zeros(300)
                for word1 in splt_body:
                    count_words_1 += 1
                    if word1 in model:
                        vectors_1 = np.add(vectors_1, model[word1])
                    else:
                        vectors_1 = np.add(vectors_1, np.zeros(300))

                each_post['body'] = np.divide(vectors_1, count_words_1)

            else:
                each_post['body'] = np.zeros(300)

    types = []
    distances = []
    for post in posts:
        for i in range(len(post)):
            try:
                ans_vector = post[i].get('body')
                if i != 0:
                    types.append(post[i].get('majority_type'))
                if 'post_depth' in post[i]:
                    depth = post[i].get('post_depth')
                    if depth != None:
                        que_vector = post[depth - 1].get('body')
                        distances.append(np.concatenate((que_vector, ans_vector)))
                    else:
                        que_vector = post[0].get('body')
                        distances.append(np.concatenate((que_vector, ans_vector)))
            except:
                que_vector = post[0].get('body')
                distances.append(np.concatenate((que_vector, ans_vector)))

    types_1 = []
    distances_1 = []
    for i in range(len(types)):
        if types[i] != None:
            types_1.append(types[i])
            distances_1.append(distances[i])

    X_1 = np.asarray(distances_1)
    y_1 = np.asarray(types_1)

    return X_1, y_1