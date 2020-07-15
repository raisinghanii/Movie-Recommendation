import pickle
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz
from flask import Flask, request, render_template
from scipy.spatial.distance import cosine
from surprise import SVD
import random

# load data
movie_user_mat_sparse = pickle.load(open('movie_user_mat_sparse', 'rb'))
movie_to_idx = pickle.load(open('movie_to_idx', 'rb'))
model = pickle.load(open('model_svd_100','rb'))
trainset = pickle.load(open('trainset','rb'))
mname = pickle.load(open('moviename','rb'))

# fit knn
model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=50, n_jobs=-1)
model_knn.fit(movie_user_mat_sparse)


def fuzzy_2(favs,m = mname):
    final = []
    for i in favs:
        lst = []
        for j in m:
            ratio = fuzz.ratio(i.lower(),j.lower())
            if ratio>=70:
                lst.append([j,ratio])
                
        list.sort(lst,key = lambda x: x[1],reverse = True)
        final.append(lst[0][0])
    return final    


def fuzzy_matching(mapper, fav_movie, verbose=True):
    match_tuple = []
    
    for title, idx in mapper.items():
        ratio = fuzz.ratio(title.lower(), fav_movie.lower())
        if ratio >= 70:
          
            match_tuple.append((title, idx, ratio))
    match_tuple = sorted(match_tuple, key=lambda x: x[2])[::-1]
    if not match_tuple:
        print('Oops! No match is found')
        return
    if verbose:
        print('Found possible matches in our database: {0}\n'.format([x[0] for x in match_tuple]))
    return match_tuple[0][1]


def fuzzy_matching_2(mapper, fav_movie, verbose=True):

    match_tuple = []

    for i in mapper:
        ratio = fuzz.ratio(i.lower(),fav_movie.lower())
        if ratio>=70:
            match_tuple.append((i,ratio))

    match_tuple = sorted(match_tuple, key=lambda x: x[1])[::-1]
    if not match_tuple:
        return "00"
    if verbose:
        return match_tuple[0][0]



def removecommon(cf):
    final = []
    list.sort(cf)
    for i in range(len(cf)-1):
        ratio = fuzz.ratio(cf[i].lower(),cf[i+1].lower())
        if ratio>=70:
            final.append(cf[i])
            
    return final


# knn code
def make_recommendation(fav_movies, model_knn=model_knn, data=movie_user_mat_sparse, mapper=movie_to_idx,n_recommendations=20):
    final = []
    result = []
    for fav_movie in fav_movies:
        print('You have input movie:', fav_movie)
        idx = fuzzy_matching(mapper, fav_movie, verbose=True)
    
        print('Recommendation system start to make inference')
        print('......\n')
        distances, indices = model_knn.kneighbors(data[idx], n_neighbors=n_recommendations + 1)
        
        raw_recommends = \
            sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())), key=lambda x: x[1])[:0:-1]

        for j in raw_recommends:
            result.append([j[0], j[1]])

    reverse_mapper = {v: k for k, v in mapper.items()}
    
    result = sorted(result, key=lambda x: x[1])
    
    for i in result:
        final.append(reverse_mapper[i[0]])
    return final


#svd code

def display(similarity_table):
    similarity_table = pd.DataFrame(
        similarity_table,
        columns=['vector cosine distance', 'movie title']
    ).sort_values('vector cosine distance', ascending=True)
    
    return similarity_table.iloc[:20]


def get_vector_by_movie_title(movie_title: str, trained_model: SVD) -> np.array:
    movie_row_idx = trainset._raw2inner_id_items[movie_title]
    return trained_model.qi[movie_row_idx]


def cosine_distance(vector_a: np.array, vector_b: np.array) -> float:
    return cosine(vector_a, vector_b)


def get_top_similarities(movie_titles, model: SVD) -> pd.DataFrame:
    similarity_table = []
    
    for movie_title in movie_titles:
        
        movie_vector: np.array = get_vector_by_movie_title(movie_title, model)
        #print("latent features ", movie_vector)

        for other_movie_title in model.trainset._raw2inner_id_items.keys():
            
            other_movie_vector = get_vector_by_movie_title(other_movie_title, model)

            similarity_score = cosine_distance(other_movie_vector, movie_vector)
            
            similarity_table.append((similarity_score, other_movie_title))

    return display(sorted(similarity_table))




#content based code
def get_title_from_index(df,index):
    return df[df.index == index]["title"].values[0]


def get_index_from_title(df,title):
    return df[df.title == title]["index"].values[0]


def combine_features(row):
    try:
        return row['keywords'] + " " + row['cast'] + " " + row["genres"] + " " + row["director"]
    except:
        print("Error:", row)
        
def calcsim():
    df = pd.read_csv("dataset.csv")

    features = ['keywords', 'cast', 'genres', 'director']

    for feature in features:
        df[feature] = df[feature].fillna('')

    x = df.apply(combine_features, axis=1)

    cv = CountVectorizer()

    count_matrix = cv.fit_transform(x)

    cosine_sim = cosine_similarity(count_matrix)

    return df,cosine_sim


def suggest(movie_user_likes):
    try:
        df.head()
        cosine_sim.shape
    except:
        df,cosine_sim = calcsim()


    movie_user_like = fuzzy_matching_2(df['title'],movie_user_likes)
    print("movie likes = ",movie_user_like)

    movie_index = get_index_from_title(df,movie_user_like)

    similar_movies = list(enumerate(cosine_sim[movie_index]))

    sorted_similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)

    movies = []

    i = 0
    for element in sorted_similar_movies:
        # print(get_title_from_index(element[0]))
        movies.append(get_title_from_index(df,element[0]))
        i = i + 1
        if i > 50:
            break

    return movies


def contb(favs_list):
    lst = []
    for i in favs_list:
        try:
            l  = suggest(i)
            cnt=0
            for j in l:
                lst.append(j)
                cnt = cnt  + 1
                if cnt == 15:break
        except:
            pass
    
    return lst


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        int_features = [str(x) for x in request.form.getlist('movie[]')]
    
        int_feat = fuzzy_2(int_features)
    
        mylist = []
        res = make_recommendation(int_features)
        res2 = get_top_similarities(int_feat,model).values.tolist()
        res3 = []
        try:
            res3 = contb(int_features)
        except:
            pass
        
        res2 = [item[1] for item in res2]
    
        for i in res:
            mylist.append(i)
        for i in res2:
            mylist.append(i)
        for i in res3:
            mylist.append(i)
    
        mylist = list(dict.fromkeys(mylist))
        
        rem = removecommon(mylist)
    
        random.shuffle(mylist)
        
        mylist = list(set(mylist).difference(set(int_feat)))
        mylist = list(set(mylist).difference(set(rem)))
        
        random.shuffle(mylist)
    
        return render_template('2ndpage.html',r=mylist)
    except :
        return render_template('notfound.html')


if __name__ == "__main__":
    app.run(host="localhost", port=int("777"))