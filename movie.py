import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv('g_score.csv')
#print(df.info())
#print(df.isnull().sum())
#print(df[df.isnull().any(axis=1)])
movie_matrix = df.pivot_table(index="movieId",columns="tagId",values="relevance",fill_value=0)
similarity = cosine_similarity(movie_matrix)

#function to return recommendation
def recommned(id,similarity,movie_matrix,number=5):
    movie_idx = movie_matrix.index.get_loc(id)
    scores = list(enumerate(similarity[id]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    top_recommendations = [int(movie_matrix.index[i[0]]) for i in scores[1:number + 1]]
    return top_recommendations

recommendations = recommned(id=1, similarity=similarity,movie_matrix=movie_matrix )
print("Recommended movies:", recommendations)
#this gives out the movieID as results