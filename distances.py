import gzip
import pickle
import numpy as np


def generate_matrixes():
    """
    This function computes the matrixes with increasing depth search
    from the matrix with depth search 1, and saves the resulting matrixes
    to data/dist{n}.gz
    """
    f = gzip.open("./data/dist1.gz", "rb")
    dist, x, y = pickle.load(f)
    for i in range(x):
        for j in range(y):
            if dist[i, j] > 1:
                dist[i, j] = 1
    f = gzip.open("./data/dist1.gz", "w")
    pickle.dump((dist, x, y), f)

    for n in range(9):
        f = gzip.open("./data/dist" + str(n+1) + ".gz", 'rb')
        dist, x, y = pickle.load(f)
        count = 0
        for i in range(x):
            for j in range(y):
                if dist[i, j] != 0:
                    count += 1

        print("pour n = " + str(n+1))
        print(count)

        newdist = np.identity(x)
        for i in range(x):
            for j in range(i+1, y, 1):
                maxvalue = 0
                if dist[i, j] != 0:
                    maxvalue = dist[i, j]
                else:
                    for k in range(x):
                        if dist[i, k] * dist[k, j] != 0:
                            value = 1/((1/dist[i, k]) + (1/dist[k, j]))
                            if value > maxvalue:
                                maxvalue = value
                newdist[i, j] = maxvalue
                newdist[j, i] = maxvalue
        f = gzip.open("./data/dist" + str(n+2) + ".gz", "w")
        pickle.dump((newdist, x, y), f)


def cosine_similarity(movie_matrix, user_vector, i):
    """
    This function computes the cosine similarity between a user vector and a movie vector
    The movie vector is the transpose of the i-th line of the movie_matrix
    :param movie_matrix: the matrix containing the movie vectors
    :param user_vector: the vector representation of a user
    :param i: the index of the movie vector in the movie matrix
    :return: the resulting cosine similarity
    """
    return np.dot(movie_matrix[:, i], user_vector) / (np.linalg.norm(user_vector) * np.linalg.norm(movie_matrix[:, i]))


def extract_movie_matrix(level):
    """
    This function extracts the sub-matrix of size 252*252 from a larger matrix also
    describing other resources like actors, directors and movie genres. Since for the user-movie
    matching, we are only interested in the movie representations, we can reduce the dimension to 252.
    :param level: the level of depth of the connection search between resources
    :return: the extracted matrix
    """
    f = gzip.open("./data/mapping.gz", 'rb')
    mapping = pickle.load(f)
    size = len(mapping)

    f = gzip.open("./data/dist" + str(level) + ".gz", 'rb')
    matrix = pickle.load(f)[0]
    return matrix[0:size, 0:size]

