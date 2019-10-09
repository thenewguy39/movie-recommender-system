##Modules imported

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

##Datasets imported

ratings="./data/ratings.csv"
movies="./data/movies.csv"
df_ratings = pd.read_csv(ratings, sep=',')
df_ratings.columns = ['userId', 'itemId', 'rating', 'timestamp']
df_movies = pd.read_csv(movies, sep=',')
df_movies.columns = ['itemId', 'title', 'genres']
#print(df_movies.head())
#print(df_ratings.head())
#print(df_ratings.describe())


##Creating user-item dataset by pivoting df_ratings

df_user_item = df_ratings.pivot(index = 'userId' , columns = 'itemId' , values = 'rating' ) 
""" Sort index/rows (userId's) and columns (itemId's)"""
df_user_item.sort_index(axis=0, inplace=True)
df_user_item.sort_index(axis=1, inplace=True)
#print(df_user_item.head())
#print(df_user_item.loc[1][:10]) #rating of userId = 1 for itemId = 1 to 10

##Approximate Singular Value Decomposition using Stochastic Gradient Descent
"""
In SVD:
We decompose a matrix R into the best lower rank (i.e. smaller/simpler) approximation of the
original matrix  R.Mathematically, it decomposes R into a two unitary matrices and a diagonal matrix:
R=UΣVT 
where:
R is users's ratings matrix,
U  is the user "features" matrix, it represents how much users "like" each feature,
Σ  is the diagonal matrix of singular values (essentially weights),
VT  is the movie "features" matrix, it represents how relevant each feature is to each movie,
with  U  and  VT  orthogonal.
In SGD:
We inject  Σ  into U and V.
Then we try to find P and q such that  Rˆ=PQT  is close to  R  for the item-user pairs already rated.

"""

def encode_ids(data):
    """Takes a rating dataframe and return: 
    1. a simplified rating dataframe with ids in range(nb unique id) for users and movies
    2. two mapping dictionaries
    
    """

    data_encoded = data.copy()
    
    users = pd.DataFrame(data_encoded.userId.unique(),columns=['userId'])  # df of all unique users
    dict_users = users.to_dict()    
    inv_dict_users = {v: k for k, v in dict_users['userId'].items()}

    items = pd.DataFrame(data_encoded.itemId.unique(),columns=['itemId']) # df of all unique items
    dict_items = items.to_dict()    
    inv_dict_items = {v: k for k, v in dict_items['itemId'].items()}

    data_encoded.userId = data_encoded.userId.map(inv_dict_users)
    data_encoded.itemId = data_encoded.itemId.map(inv_dict_items)

    return data_encoded, dict_users, dict_items


def SGD(data,           # dataframe containing 1 (user|item) rating per row
        n_factors = 10, # number of factors
        alpha = .01,    # number of factors
        n_epochs = 5,   # number of iteration of the SGD procedure
       ):
    """
    Procedure to implement SGD():
    1.   initialize P and Q to random values
    2.   for n epochs(passes) on the data:
             for all known ratings r_ui
                 compute the error between the predicted rating (p[u]).(q[i]) and the known ratings r_ui:
                     err = r_ui - (p[u]).(q[i])
                 update p[u] and q[i] with the following rule:
                     p[u] <- p[u] + alpha * err * q[i]  
                     q[i] <- q[i] + alpha * err * p[u]
    """

    ## Encoding userId's and itemId's in data
    data, dict_users, dict_items = encode_ids(data)
    
    n_users = data.userId.nunique()  # number of unique users
    n_items = data.itemId.nunique()  # number of unique items
    
    ## Randomly initialize the user and item vectors.
    p = np.random.normal(0, .1, (n_users, n_factors))
    q = np.random.normal(0, .1, (n_items, n_factors))

    ## Optimization procedure
    for epoch in range(n_epochs):
        print ('epoch: ', epoch)
        ## Loop over the rows in data
        for index in range(data.shape[0]):
            row = data.iloc[[index]]
            u = int(row.userId)      # current userId = position in the p vector (thanks to the encoding)
            i = int(row.itemId)      # current itemId = position in the q vector
            r_ui = float(row.rating) # rating associated to the couple (user u , item i)
            
  
            err = r_ui - np.dot(p[u],(q[i]).transpose())
            
            ## Update vectors p[u] and q[i]
            p_old = p[u]
            p[u] = p[u] + alpha * err * q[i]   
            q[i] = q[i] + alpha * err * p_old
            
    return p, q
    
    
def estimate(u, i, p, q):
    """Estimate rating of user u for item i."""
    return np.dot(p[u],(q[i]).transpose())

p, q = SGD(df_ratings)

##Approximate user-item dataset filled with predicted ratings

df_user_item_filled = pd.DataFrame(np.dot(p, q.transpose()))
#print(df_user_item_filled.head())

##Reverting to the original id's from the encoded id's

df_ratings_encoded, dict_users, dict_items = encode_ids(df_ratings)
df_user_item_filled.rename(columns=(dict_items['itemId']), inplace=True)
df_user_item_filled.rename(index=(dict_users['userId']), inplace=True)
## Sort index/rows (userId's) and columns (itemId's)
df_user_item_filled.sort_index(axis=0, inplace=True)
df_user_item_filled.sort_index(axis=1, inplace=True)
#print(df_user_item_filled.head())

#print(df_user_item.loc[1][:10]) #Actual ratings of userId=1
#print(df_user_item_filled.loc[1][:10]) #Predicted ratings of userId=1

##Top ten movies as rated by userId=10

rated = list((df_user_item.loc[10]).sort_values(ascending=False)[:10].index)
#print(rated)
print("")
print("TOP 10 MOVIES AS RATED BY USER 10 :")
print("")
print(df_movies[df_movies.itemId.isin(rated)][['title','genres']])

##Top ten movie recommendations to userId=10

recommendations = list((df_user_item_filled.loc[10]).sort_values(ascending=False)[:10].index)
#print(recommendations)
print("")
print("MOVIE RECOMMENDATIONS FOR USER 10 :")
print("")
print(df_movies[df_movies.itemId.isin(recommendations)][['title','genres']])



##Plot of actual rating v/s predicted rating of Movie 1 by Users 1 to 100

ax = df_user_item_filled.reset_index()[:100].plot( x = 'index' , y = 1 , color = 'red' ,kind='scatter')
p2 = df_user_item.reset_index()[:100].plot( ax = ax , x = 'userId', y = 1 , kind = 'scatter')
plt.ylim(bottom = 0.1)
plt.title("Actual(Blue) v/s Predicted(Red) ratings for 'Toy Story' by users 1 to 100")
plt.ylabel("Rating")
plt.xlabel("User ID")
plt.show()
