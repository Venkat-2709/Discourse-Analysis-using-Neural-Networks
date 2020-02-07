def set_data(data):
    posts = []
    subreddit = []

    for dicti in data:
        post = dicti.get('posts')
        subredit = dicti.get('subreddit')
        subreddit.append(subredit)
        posts.append(post)

    subreddit = subreddit
    posts = posts

    return subreddit, posts


def clean_data(tokenizer, lemmatizer, subreddit, posts):
    sub_annotations = []
    all_bodies = []

    for post in posts:
        annotations = []
        bodies = []
        for comment in post:
            annotation = comment.get('majority_type')
            annotations.append(annotation)
            if 'author' in comment:
                del comment['author']
            if 'annotations' in comment:
                del comment['annotations']
            body = comment.get('body')
            if body != None:
                tokens = tokenizer.tokenize(body)
                lem_body = ' '.join([lemmatizer.lemmatize(word) for word in tokens])
                comment['body'] = lem_body
            body = comment.get('body')
            bodies.append(body)

        sub_annotations.append(annotations)
        all_bodies.append(bodies)

    return sub_annotations, all_bodies, subreddit, posts