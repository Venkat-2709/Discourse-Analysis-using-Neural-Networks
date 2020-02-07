# Discourse-Analysis-using-Neural-Networks
## Discourse Analysis to determine the nature of the comments with respective to the parent comment.

## Step 1: Setting up the project
```
Download the Word2vec from https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit 
```

## Step 2: Importing the Libraries
```
NLTK 3.4.5 -----> pip install nltk (or) conda install nltk [For anaconda users]
Numpy 1.18.1 -----> pip install numpy (or) conda install numpy
Gensim ----> pip install gensim (or) conda install gensim
```

## Step 3: Installing TensorFlow
```
For installing TensorFlow Go to:https://www.tensorflow.org/install and install appropriate version.
```

## Step 4: Downloading Dataset
```
Download https://www.kaggle.com/rtatman/discourse-acts-on-reddit
```

## Step 5: Converting .csv to .json

Split the dataset into train and test set and then do the following to convert the csv to json file
```
import csv
import json

with open('filename.csv') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

with open('filename.json', 'w') as f:
    json.dump(rows, f)
```

## Step 6: Execute
```
Execute the main.py to train the model and then use the model to predict the result.
```
