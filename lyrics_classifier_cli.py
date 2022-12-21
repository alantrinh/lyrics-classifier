import argparse
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('-l', '--lyric', help='song lyric to predict')
args = parser.parse_args()

with open('lyrics_classifier.pkl', 'rb') as pickle_file:
    pipeline = pickle.load(pickle_file)

with open('lyrics_classifier_vectorizer.pkl', 'rb') as pickle_file:
    vectorizer = pickle.load(pickle_file)

if args.lyric:
    new_vectors = vectorizer.transform([args.lyric])
    prediction = pipeline.predict(new_vectors)
    probability = pipeline.predict_proba(new_vectors)
    print(prediction, probability)