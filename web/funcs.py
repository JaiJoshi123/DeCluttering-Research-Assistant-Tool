import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import bs4 as bs
import urllib.request
import re
import nltk
import heapq

def createdf():
    df = pd.read_csv('C:/Users/Jai Joshi/Documents/browser_buddy/web/user_item_interactions.csv')
    # df_content = pd.read_csv('/articles_community.csv')
    del df['Unnamed: 0']
    # del df_content['Unnamed: 0']
    coded_dict = dict()
    cter = 1
    email_encoded = []
    
    for val in df['email']:
        if val not in coded_dict:
            coded_dict[val] = cter
            cter+=1
        
        email_encoded.append(coded_dict[val])
    df['user_id'] = email_encoded
    return df

def create_user_item_matrix(df):
    user_item = df.groupby(['user_id', 'article_id']).count()['title'].unstack()
    for col in user_item.columns:
        user_item[col] = user_item[col].apply(lambda x: 1 if (x>0) else 0)
    
    return user_item # return the user_item matrix 

# user_item = create_user_item_matrix(df)
def getrecomm(user_id, df,  m):
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

def user_user_recs(user_id, df, m):
    user_item = df.groupby(['user_id', 'article_id']).count()['title'].unstack()
    for col in user_item.columns:
        user_item[col] = user_item[col].apply(lambda x: 1 if (x>0) else 0)
    
    recs = [] #article recommendation list
    #find similar users to user_id
    similar_users = find_similar_users(user_id, df)
    
    #get recommendations
    for user in similar_users:
        new_recs = np.setdiff1d(get_user_articles(user_id, df, user_item)[0], get_user_articles(user, df, user_item)[0])
        recs.extend(np.setdiff1d(new_recs, recs))
        if len(recs) >= m:
            break
    
    return recs[:m]

def find_similar_users(user_id, df):
    user_item = df.groupby(['user_id', 'article_id']).count()['title'].unstack()
    for col in user_item.columns:
        user_item[col] = user_item[col].apply(lambda x: 1 if (x>0) else 0)
    compute_sim = np.array(user_item.loc[user_id, :]).dot(np.transpose(user_item))
    sorted_sim = pd.Series(data = compute_sim, index = user_item.index).sort_values(ascending = False)
    most_similar_users = list(sorted_sim.index)
    most_similar_users.remove(user_id)
    
    return most_similar_users

def get_user_articles(user_id, df, user_item):
    article_ids = list(user_item.T[user_item.loc[user_id, :]>0].T.columns.astype('str'))
    article_names = get_article_names(article_ids, df)
    return article_ids, article_names

def get_article_names(article_ids, df):
    article_names = []
    for val in article_ids:
        article_names.append(df[df['article_id']== float(val)]['title'].unique()[0])
        
    return article_names


def get_summary(url):
    raw_data = urllib.request.urlopen(url) 
    document = raw_data.read()
    parsed_document = bs.BeautifulSoup(document,'lxml')
    article_paras = parsed_document.find_all('p')
    links = parsed_document.find_all('a')
    scrapped_data = ""
    for para in article_paras:
        scrapped_data += para.text
    
    scrapped_data = re.sub(r'\[[0-9]*\]', ' ',  scrapped_data)
    scrapped_data = re.sub(r'\s+', ' ',  scrapped_data)
    formatted_text = re.sub('[^a-zA-Z]', ' ', scrapped_data)
    formatted_text = re.sub(r'\s+', ' ', formatted_text)

    all_sentences = nltk.sent_tokenize(scrapped_data)
    stopwords = nltk.corpus.stopwords.words('english')
    word_freq = {}
    for word in nltk.word_tokenize(formatted_text):
        if word not in stopwords:
            if word not in word_freq.keys():
                word_freq[word] = 1
            else:
                word_freq[word] += 1
    max_freq = max(word_freq.values())
    for word in word_freq.keys():
        word_freq[word] = (word_freq[word]/max_freq)
    
    sentence_scores = {}
    for sentence in all_sentences:
        for token in nltk.word_tokenize(sentence.lower()):
            if token in word_freq.keys():
                if len(sentence.split(' ')) <25:
                    if sentence not in sentence_scores.keys():
                        sentence_scores[sentence] = word_freq[token]
                    else:
                        sentence_scores[sentence] += word_freq[token]
    
    selected_sentences= heapq.nlargest(5, sentence_scores, key=sentence_scores.get)
    text_summary = ' '.join(selected_sentences)

    h1 = parsed_document.find('h1', {"id": "firstHeading"})
    title = h1.find('span').text
    return (title,text_summary)

def get_summary_2(scrapped_data):
    scrapped_data = re.sub(r'\[[0-9]*\]', ' ',  scrapped_data)
    scrapped_data = re.sub(r'\s+', ' ',  scrapped_data)
    formatted_text = re.sub('[^a-zA-Z]', ' ', scrapped_data)
    formatted_text = re.sub(r'\s+', ' ', formatted_text)

    all_sentences = nltk.sent_tokenize(scrapped_data)
    stopwords = nltk.corpus.stopwords.words('english')
    word_freq = {}
    for word in nltk.word_tokenize(formatted_text):
        if word not in stopwords:
            if word not in word_freq.keys():
                word_freq[word] = 1
            else:
                word_freq[word] += 1
    max_freq = max(word_freq.values())
    for word in word_freq.keys():
        word_freq[word] = (word_freq[word]/max_freq)
    
    sentence_scores = {}
    for sentence in all_sentences:
        for token in nltk.word_tokenize(sentence.lower()):
            if token in word_freq.keys():
                if len(sentence.split(' ')) <25:
                    if sentence not in sentence_scores.keys():
                        sentence_scores[sentence] = word_freq[token]
                    else:
                        sentence_scores[sentence] += word_freq[token]
    
    selected_sentences= heapq.nlargest(5, sentence_scores, key=sentence_scores.get)
    text_summary = ' '.join(selected_sentences)

    return text_summary