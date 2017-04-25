import csv
import pickle
import gzip
import urllib.parse
import numpy as np
import distances
from sklearn.utils import shuffle


def export_rated_movies():
    """
    This function selects the dbpedia uris corresponding to movies with ratings,
    and exports the resulting list in the file data/dbpedia_rated_uris.gz
    """
    movielens_ids = []
    dbpedia_uris = []
    with open('./data/ratings.csv', 'rt', encoding='utf8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in reader:
            movie_id = row[1]
            if movie_id not in movielens_ids:
                movielens_ids.append(movie_id)

    with open('./data/MappingMovielens2DBPedia-1.2.tsv', 'rt', encoding='utf8') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t', quotechar='|')
        for row in reader:
            movielens_id = row[0]
            dbpedia_uri = row[2]
            if movielens_id in movielens_ids:
                dbpedia_uris.append(dbpedia_uri)

    try:
        f = gzip.open('./data/dbpedia_rated_uris.gz', 'w')
        pickle.dump(dbpedia_uris, f)
    except:
        print("An error occured while saving the dbpedia URIs")


def create_movielens_mapping():
    """
    This function filters and orders the movielens_ids to map the list of dbpedia uris contained in /data/mapping.gz,
    it exports the resulting list in the file data/movielens_ids.gz
    """

    # mapped dbpedia uris
    f = gzip.open("./data/mapping.gz", 'rb')
    mapping = pickle.load(f)

    movielens_ids = []
    dbpedia_uris = []
    with open('./data/MappingMovielens2DBPedia-1.2.tsv', 'rt', encoding='utf8') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        for row in reader:
            movielens_ids.append(row[0])
            dbpedia_uris.append(row[2])

    filtered_movielens_ids = []
    for uri in mapping:
        filtered_movielens_ids.append(movielens_ids[dbpedia_uris.index(urllib.parse.unquote(uri))])

    print(len(filtered_movielens_ids))
    f = gzip.open("./data/movielens_ids.gz", 'w')
    pickle.dump(filtered_movielens_ids, f)


def create_user_vector(user_id, hold_out):
    """
    This function creates two user vectors from the user id and a hold out ratio
    The first vector is a vector with 0s everywhere and values computed from the ratings on the rated movies that weren't held out
    The second vector is a vector with 0s everywhere and values computed from the ratings on the held out rated movies
    :param user_id: user id from movielens movie reviews
    :param hold_out: ratio of reviewed movies to ignore during the creation of a vector representing the user
    :return: two np.arrays corresponding to the user vector and to the hold-out user vector
    """
    movielens_ids = []
    ratings = []
    with open('./data/ratings.csv', 'rt', encoding='utf8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            uid = row[0]
            if uid == str(user_id):
                movie_id = row[1]
                rating = row[2]
                ratings.append(rating)
                movielens_ids.append(movie_id)

    f = gzip.open("./data/movielens_ids.gz", 'rb')
    mapping = pickle.load(f)

    vector = np.zeros((len(mapping), 1))
    holdout_vector = np.zeros((len(mapping), 1))

    total = 0
    for ml_id in movielens_ids:
        if ml_id in mapping:
            total += 1

    # shuffle the two lists with the same permutation (from sklearn)
    ratings, movielens_ids = shuffle(ratings, movielens_ids)

    count = 0
    for ml_id in movielens_ids:
        if (ml_id in mapping) and (count < (1 - hold_out)*total):
            count += 1
            vector[mapping.index(ml_id)] = float(ratings[movielens_ids.index(ml_id)])/2.5 - 1
        elif ml_id in mapping:
            if float(ratings[movielens_ids.index(ml_id)]) > 3.5:
                holdout_vector[mapping.index(ml_id)] = 1
            if float(ratings[movielens_ids.index(ml_id)]) < 3.5:
                holdout_vector[mapping.index(ml_id)] = -1
    return vector, holdout_vector


def recommend_movies(level, user_vector):
    """
    This function computes the cosine similarity of the given user vector with all the movie vectors
    :param level: the depth of relation search in the linkedmdb graph
    :param user_vector: the vector representation of the user
    :return: a list of cosine similarities
    """
    matrix = distances.extract_movie_matrix(level)
    scores = []

    x, y = np.shape(matrix)
    for i in range(x):
        scores.append(distances.cosine_similarity(matrix, user_vector, i))

    return scores


def recommend_best_movie(level, user_vector):
    """
    This function recommends the movie with the best score, after filtering out the already rated movies
    :param level: the depth of relation search in the linkedmdb graph
    :param user_vector: the vector representation of the user
    :return: the id of the recommended movie
    """
    scores = recommend_movies(level, user_vector)
    filtered_scores = np.subtract(scores, np.abs(user_vector))

    return np.argmax(filtered_scores)


def test_user(user_id, hold_out, level, threshold):
    """
    This function tests our algorithm for a given user and given parameters.
    A way to test this algorithm is to see it as a classifier.
    If it can efficiently classify well rated movies from badly rated movies for a user, then
    it's proof that the graph distance is a semantic measure of the relatedness between movies.
    :param user_id: user id from movielens movie reviews
    :param hold_out: ratio of reviewed movies to ignore during the creation of a vector representing the user
    :param level: the depth of relation search in the linkedmdb graph
    :param threshold: the limit we use to decide whether a rating is positive or negative
    :return: the count of true positives, true negatives, false positives and false negatives for this user.
    """
    user_vector, user_holdout_vector = create_user_vector(user_id, hold_out)
    scores = recommend_movies(level, user_vector)
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    for index in range(len(user_holdout_vector)):
        has_liked = user_holdout_vector[index]
        if has_liked == 1:
            if scores[index] > threshold:
                true_positives += 1
            else:
                false_negatives += 1
        elif has_liked == -1:
            if scores[index] > threshold:
                false_positives += 1
            else:
                true_negatives += 1
    return true_positives, true_negatives, false_positives, false_negatives


def test(hold_out=0.2, level=5, threshold=0.15):
    """
    This function tests the algorithm on all the users.
    With the given amount of data, the results aren't very conclusive.
    :param hold_out: ratio of reviewed movies to ignore during the creation of the vectors representing the users
    :param level: the depth of relation search in the linkedmdb graph
    :param threshold: the limit we use to decide whether a rating is positive or negative
    :return: the count of true positives, true negatives, false positives and false negatives
    """
    total_right_positives = 0
    total_wrong_positives = 0
    total_right_negatives = 0
    total_wrong_negatives = 0
    for i in range(671):
        rp, rn, wp, wn = test_user(i, hold_out, level, threshold)
        total_right_positives += rp
        total_wrong_positives += wp
        total_right_negatives += rn
        total_wrong_negatives += wn

    total = total_right_negatives + total_right_positives + total_wrong_negatives + total_wrong_positives
    print("Accuracy : " + str(total_right_negatives + total_right_positives) + "/" + str(total))
    print("Right positives : " + str(total_right_positives))
    print("Right negatives : " + str(total_right_negatives))
    print("Wrong positives : " + str(total_wrong_positives))
    print("Wrong negatives : " + str(total_wrong_negatives))
    return total_right_positives, total_right_negatives, total_wrong_positives, total_wrong_negatives


def export_results():
    """
    Function that calls the test function for different sets of parameters,
    and exports the results in the file /data/results.csv
    """
    with open('./data/results.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['hold_out', 'level', 'threshold', 'tp', 'tn', 'fn', 'fp'])
        for holdout in np.arange(0.1, 0.5, 0.1):
            for level in np.arange(1, 6, 1):
                for threshold in np.arange(0.07, 0.25, 0.02):
                    tp, fp, tn, fn = test(holdout, level, threshold)
                    writer.writerow([holdout, level, threshold, tp, fp, tn, fn])


