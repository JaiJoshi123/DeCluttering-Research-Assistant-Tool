import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import project_tests as t
import pickle
df = pd.read_csv('/user-item-interactions.csv')
df_content = pd.read_csv('/articles_community.csv')
del df['Unnamed: 0']
del df_content['Unnamed: 0']

def create_user_item_matrix(df):
    '''
    INPUT:
    df - pandas dataframe with article_id, title, user_id columns
    
    OUTPUT:
    user_item - user item matrix 
    
    Description:
    Return a matrix with user ids as rows and article ids on the columns with 1 values where a user interacted with 
    an article and a 0 otherwise
    '''
    user_item = df.groupby(['user_id', 'article_id']).count()['title'].unstack()
    for col in user_item.columns:
        user_item[col] = user_item[col].apply(lambda x: 1 if (x>0) else 0)
    
    return user_item # return the user_item matrix 

user_item = create_user_item_matrix(df)
def ML(user_id, m):
    user_item = df.groupby(['user_id', 'article_id']).count()['title'].unstack()
    for col in user_item.columns:
        user_item[col] = user_item[col].apply(lambda x: 1 if (x>0) else 0)
    recs = [] #article recommendation list
    #find similar users to user_id
    # compute similarity of each user to the provided user
    compute_sim = np.array(user_item.loc[user_id, :]).dot(np.transpose(user_item))

    # sort by similarity
    sorted_sim = pd.Series(data = compute_sim, index = user_item.index).sort_values(ascending = False)
    
    # create list of just the ids
    most_similar_users = list(sorted_sim.index)
    
    # remove the own user's id
    most_similar_users.remove(user_id)
    
    similar_users = most_similar_users # return a list of the users in order from most to least similar
    #get recommendations
    for user in similar_users:
        article_ids1 = list(user_item.T[user_item.loc[user_id, :]>0].T.columns.astype('str'))
        article_ids2 = list(user_item.T[user_item.loc[user, :]>0].T.columns.astype('str'))
        new_recs = np.setdiff1d(article_ids1, article_ids2)
        recs.extend(np.setdiff1d(new_recs, recs))
        if len(recs) >= m:
            break
    
    rec_article_ids = recs[:m] # return your recommendations for this user_id
    rec_article_names = []
    for val in rec_article_ids:
        rec_article_names.append(df[df['article_id']== float(val)]['title'].unique()[0])
        
    return rec_article_names # Return the article names associated with list of article ids