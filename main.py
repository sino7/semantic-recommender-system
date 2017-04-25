from SPARQLWrapper import SPARQLWrapper, JSON
import numpy as np
import pickle
import gzip
import urllib.parse
from time import *

prefix = """
    PREFIX owl: <http://www.w3.org/2002/07/owl#>
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX foaf: <http://xmlns.com/foaf/0.1/>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX oddlinker: <http://data.linkedmdb.org/resource/oddlinker/>
    PREFIX map: <file:/C:/d2r-server-0.4/mapping.n3#>
    PREFIX db: <http://data.linkedmdb.org/resource/>
    PREFIX dbpedia: <http://dbpedia.org/property/>
    PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
    PREFIX dc: <http://purl.org/dc/terms/>
    PREFIX movie: <http://data.linkedmdb.org/resource/movie/>
    prefix dbo: <http://dbpedia.org/ontology/>
    prefix movie: <http://data.linkedmdb.org/resource/movie/> """
types = ["movie", "actor", "director", "genre"]
sparql = SPARQLWrapper("http://www.linkedmdb.org/sparql")


def query_resources():
    """
    This function queries and saves the uris of all the available resources in linkedmdb
    """
    for t in types:
        resources, mapping = get_resources(t)
        save_to_file(resources, "./data/" + t + ".gz")
        print(t + " : " + str(len(resources)))
        if t == "movie":
            save_to_file(mapping, "./data/mapping.gz")


def filter_movies():
    """
    This function filters the movies to keep only those for which we have user ratings
    """
    f = gzip.open("./data/dbpedia_rated_uris.gz", 'rb')
    dbpedia_movies = pickle.load(f)
    f = gzip.open("./data/mapping.gz", 'rb')
    mapped_movies = pickle.load(f)
    f = gzip.open("./data/movie.gz", 'rb')
    lmdb_movies = pickle.load(f)
    movies_filtered = []
    mapping_filtered = []
    count = 0
    for movie in mapped_movies:
        try:
            parsed_movie = urllib.parse.unquote(movie)
            if parsed_movie in dbpedia_movies:
                mapping_filtered.append(mapped_movies[count])
                movies_filtered.append(lmdb_movies[count])
        except:
            print("error decoding one of the movie names")
        count += 1

    f = gzip.open("./data/movie.gz", 'w')
    pickle.dump(movies_filtered, f)
    f = gzip.open("./data/mapping.gz", 'w')
    pickle.dump(mapping_filtered, f)
    f.close()


def build_matrix(after=0, before=0, new=True):
    """
    This function builds the distance matrix with depth search = 1
    Due to the time it takes to query all the distances, we made it possible
    to select intervals to make it in several steps
    :param after: the beginning of the interval
    :param before: the number of resources until the end. Thus, the end of the interval is total_nb_res - before
    :param new: whether we expand or overwrite the already existing matrix
    """
    res = []
    for t in types:
        res += pickle.load(gzip.open("./data/" + t + ".gz", 'rb'))

    if new:
        m = np.identity(len(res))
    else:
        m = pickle.load(gzip.open("./data/dist1.gz", 'rb'))[0]

    x, y = m.shape

    start = strptime(asctime())

    count = 0
    for i in range(after, x - before, 1):
        for j in range(y):
            if i < j:
                try:
                    d = dist(res[i], res[j])
                    m[i, j] = d
                    m[j, i] = d
                except:
                    print("an error occurred on the query number " + str(count))
                    save_to_temp_file((m, i, j), i, j)

                count += 1
                print(count, "/", (x-before-after)*(x-after+before-1)/2)

    end = strptime(asctime())
    print('matrix built in %i hours' % (end[3] - start[3]))

    save_to_file((m, x, y), "./data/dist1.gz")


def save_to_temp_file(m, i, j):
    f = gzip.open("./data/temp/dist1_" + str(i) + "_" + str(j) + ".gz", "w")
    pickle.dump(m, f)


def save_to_file(m, path):
    f = gzip.open(path, "w")
    pickle.dump(m, f)


def get_resources(type):
    """
    The query to get all the interesting resources available in linkedmdb.
    To have consistent data and to avoid having a vector space with a too high dimension,
    we only select movies for which we have a dbpedia mapping, genres that at least have 20
    of those movies each, actors that have played in at least 10 of those movies, and directors
    that have directed at least 20 of those movies.
    :param type: the type of resource to query
    """
    query = ""
    if type == "genre":
        query = """
            select
            distinct (?genre as ?resource)
            where  {
                ?film a movie:film.
                ?film owl:sameAs ?map.
                ?film movie:actor ?actor.
                ?film movie:director ?director.
                ?film movie:genre ?genre.
                FILTER(regex(str(?map), "dbpedia"))
            }
            group by ?genre
            having(count(?film) > 30)
        """
    elif type == "actor":
        query = """
            select
            distinct (?actor as ?resource)
            where  {
                ?film a movie:film.
                ?film owl:sameAs ?map.
                ?film movie:actor ?actor.
                ?film movie:director ?director.
                ?film movie:genre ?genre.
                FILTER(regex(str(?map), "dbpedia"))
            }
            group by ?actor
            having(count(?film) > 10)
        """
    elif type == "director":
        query = """
            select
            distinct (?director as ?resource)
            where  {
                ?film a movie:film.
                ?film owl:sameAs ?map.
                ?film movie:actor ?actor.
                ?film movie:director ?director.
                ?film movie:genre ?genre.
                FILTER(regex(str(?map), "dbpedia"))
            }
            group by ?director
            having(count(?film) > 30)
        """
    elif type == "movie":
        query = """
            select
            distinct (?film as ?resource) ?map
            where  {
                ?film a movie:film.
                ?film owl:sameAs ?map.
                ?film movie:actor ?actor.
                ?film movie:director ?director.
                ?film movie:genre ?genre.
                FILTER(regex(str(?map), "dbpedia"))
            }
        """
    else:
        raise Exception('Unknown type')

    sparql.setQuery(prefix + query)
    sparql.setReturnFormat(JSON)

    results = sparql.query().convert()

    resource, mapping = [], []

    for result in results["results"]["bindings"]:
        resource.append(result["resource"]["value"])
        if type == "movie":
            mapping.append(result["map"]["value"])
    return resource, mapping


def dist(uri1, uri2):
    """
    This function makes a sparql query to find whether two resources are directly connected
    :param uri1: uri of the first resource
    :param uri2: uri of the second resource
    :return: 0 if there is no connection, >0 otherwise
    """
    sparql.setQuery(prefix + """
        select
        ?relation
        where {{<"""+uri1+"""> ?relation <"""+uri2+""">}
        union  {<"""+uri2+"""> ?relation <"""+uri1+""">}}
    """)
    sparql.setReturnFormat(JSON)

    try:
        ret = sparql.query()
    except:
       raise Exception("Erreur dans la requete")

    result = ret.convert()

    return len(result["results"]["bindings"])
