import requests
import pandas as pd
from tqdm import tqdm
from itertools import combinations
from sklearn.metrics.pairwise import cosine_distances, haversine_distances
from math import radians
from nltk.corpus import stopwords
import re
from nltk.tokenize.casual import TweetTokenizer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from time import sleep
def bearer_oauth(r, bearer_token):
    """
    Method required by bearer token authentication.
    """

    r.headers["Authorization"] = f"Bearer {bearer_token}"
    r.headers["User-Agent"] = "v2RecentSearchPython"
    return r
def get_timeline(api, user_id, max_results=1000, verbose=False, tweet_fields=['id', 'text', 'created_at', 'author_id', 'referenced_tweets'], exclude_fields=['retweets']):
    pagination_token = None
    timeline = []
    while(len(timeline) < max_results):
        posts = api.get_timelines(user_id=user_id, max_results=100, tweet_fields=tweet_fields, exclude=exclude_fields)
        timeline.extend(posts.data)
        pagination_token = posts.meta.next_token
        if(pagination_token is None):
            break
        if(verbose):
            print(f'fetched {len(posts.data)} posts; {len(timeline)} total')
    # clean up
    timeline = pd.DataFrame(timeline)
    return timeline
RATE_LIMIT_STATUS = 429
MIN_SLEEP_TIME = 15 * 60 # 15 mins is rate limit time
QUERY_SLEEP_TIME = 1.5
def get_tweet_replies(tweet_id, bearer_token, 
                      tweet_fields=['id','author_id','text','conversation_id','created_at','in_reply_to_user_id'], 
                      user_fields=['description', 'location', 'name', 'username'],
                      max_results=100, verbose=False):
    reply_search_url = 'https://api.twitter.com/2/tweets/search/all'
    auth_func = lambda x: bearer_oauth(x, bearer_token)
    replies = []
    pagination_token = None
    while(len(replies) < max_results):
        reply_query_params = {
            'query' : f'conversation_id:{tweet_id}', 
            'tweet.fields' : ','.join(tweet_fields), 
            'user.fields' : ','.join(user_fields), 
            'expansions' : 'author_id',
            'next_token' : pagination_token, 
            'max_results' : max_results
        }
        try:
            reply_response_obj = requests.get(reply_search_url, params=reply_query_params, auth=auth_func)
            reply_response = reply_response_obj.json()
            sleep(QUERY_SLEEP_TIME) # min sleep time is 1 sec
#             print(f'reply response = {reply_response}')
            if('status' in reply_response and reply_response['status']==RATE_LIMIT_STATUS):
                print(f'hit rate limit; sleeping for {MIN_SLEEP_TIME} sec')
                sleep(MIN_SLEEP_TIME)
                continue
            elif(reply_response['meta']['result_count'] > 0):
                if('data' in reply_response):
    #                 if(verbose):
    #                     print(f'{len(reply_response["data"])} data mined; {len(replies)} total')
                    # user data
                    tweet_data = reply_response['data']
                    user_data = reply_response['includes']['users']
                    tweet_user_data = []
                    for d1, d2 in zip(tweet_data, user_data):
                        d3 = d1.copy()
                        d3.update(d2)
                        tweet_user_data.append(d3)
                    replies.extend(tweet_user_data)
                else:
                    print(f'bad reply response data = {reply_response}')
                    break
                if('next_token' in reply_response['meta']):
                    pagination_token = reply_response['meta']['next_token']
                else:
                    break
            else:
                break
        except Exception as e:
            print(f'ending replies because error {e}')
            print(f'bad response object {reply_response_obj}')
    # clean up
    replies = pd.DataFrame(replies)
    return replies

def get_user_tweets_and_replies(user_name, bearer_token, api, max_timeline_tweets=100, max_reply_tweets=25, verbose=False):
    user_data = api.get_user(username=user_name)
    user_id = user_data.data.id
    user_timeline = get_timeline(api, user_id, max_results=max_timeline_tweets)
    user_timeline_replies = []
    unique_ids = user_timeline.loc[:, 'id'].unique()
    for i, id_i in tqdm(enumerate(unique_ids), total=len(unique_ids)):
#         if(verbose):
#             print(f'mining tweets for id={id_i}')
        replies_i = get_tweet_replies(id_i, bearer_token, max_results=max_reply_tweets, verbose=verbose)
        user_timeline_replies.append(replies_i)
#         if(verbose and (i+1) % 10 == 0):
#             print(f'collected {len(user_timeline_replies)} for {i+1}/{user_timeline.shape[0]}')
    user_timeline_replies = pd.concat(user_timeline_replies, axis=0)
    return user_timeline, user_timeline_replies

def extract_location_data(location):
    location_feat = location.geojson['features']
    if(len(location_feat) > 0):
        feat_properties = location_feat[0]['properties']
        conf = feat_properties['accuracy']
        country = feat_properties.get('country')
        state = feat_properties.get('state')
        city = feat_properties.get('city')
        lat = feat_properties.get('lat')
        lon = feat_properties.get('lng')
        return conf, country, state, city, lat, lon
    else:
        return (None,)*6
    
EARTH_RADIUS = 6371000/1000  # multiply by Earth radius to get kilometers
def compute_divergence(reply_users, user_data, user_data_type='description'):
    # convert user description/location to vector
    # compute divergence between users in replies
    user_divergence = []
    for user_i, user_j in combinations(reply_users, 2):
        if(user_data_type == 'description'):
            dist_i_j = cosine_distances(user_data.loc[[user_i], :],
                                        user_data.loc[[user_j], :])[0][0]
        elif(user_data_type == 'location'):
            # discrete locations
#             dist_i_j = (user_data.loc[user_i, :]==user_data.loc[user_j, :]).astype(int)
            # location lat/lon
            dist_i_j = haversine_distances(user_data.loc[[user_i], :].apply(lambda x: pd.Series(map(radians, x)), axis=1),
                                           user_data.loc[[user_j], :].apply(lambda x: pd.Series(map(radians, x)), axis=1))[0][0] * EARTH_RADIUS
        user_divergence.append([user_i, user_j, dist_i_j])
    user_divergence = pd.DataFrame(user_divergence, columns=['user1','user2','dist'])
    ## compute mean pairwise divergence
    mean_user_divergence = user_divergence.loc[:, 'dist'].mean()
    return mean_user_divergence
PUNCT = list(';:,>?!.()[]/\\"\'*@')
USER_MATCHER = re.compile('@\w+')
TXT_REPLACERS = [
    [USER_MATCHER, '@USER'],
]
STOPS = stopwords.words('english') + PUNCT
def compute_log_odds(data, text_var, group_var, word_categories=None):
    group_vals = data.loc[:, group_var].dropna().unique()
    group_word_counts = []
    tokenizer = TweetTokenizer()
    for i, (group_val_i, data_i) in enumerate(data.groupby(group_var)):
        # get word counts
        cv = CountVectorizer(min_df=0., max_df=1., tokenizer=tokenizer.tokenize, stop_words=STOPS)
        clean_txt = data_i.loc[:, text_var]
        for matcher, sub in TXT_REPLACERS:
            clean_txt = clean_txt.apply(lambda x: matcher.sub(sub, x))
        dtm = cv.fit_transform(clean_txt)
        sorted_vocab = list(sorted(cv.vocabulary_.keys(), key=cv.vocabulary_.get))
        word_counts_i = pd.DataFrame([np.array(dtm.sum(axis=0))[0]], index=[group_val_i]).transpose()
        # normalize
#         word_counts_i = word_counts_i / word_counts_i.sum()
        # align vocab
        word_counts_i.index = sorted_vocab
        # optional: aggregate word categories
        if(word_categories is not None):
            word_category_counts_i = []
            for cat_j, matcher_j in zip(word_categories.index, word_categories):
                vocab_j = list(filter(lambda x: matcher_j.match(x), sorted_vocab))
                if(len(vocab_j) > 0):
                    count_j = word_counts_i.loc[vocab_j].sum()
                    word_category_counts_i.append([cat_j, count_j])
            cats_i, counts_i = zip(*word_category_counts_i)
            word_counts_i = pd.DataFrame(counts_i, index=cats_i, columns=[group_val_i])
        group_word_counts.append(word_counts_i)
    group_word_counts = pd.concat(group_word_counts, axis=1)
    # fill na
    group_word_counts.fillna(0., inplace=True)
    # smooth for log
    smooth_val = 1
    group_word_counts += smooth_val
    # normalize
    for group_val_i in group_vals:
        group_word_counts = group_word_counts.assign(**{
            group_val_i : group_word_counts.loc[:, group_val_i].values / group_word_counts.loc[:, group_val_i].sum()
        })
#     smooth_val = 1e-5
#     group_word_counts += smooth_val
    group_val_1, group_val_2 = group_vals
    word_ratio = group_word_counts.loc[:, group_val_1] / group_word_counts.loc[:, group_val_2]
    word_ratio.dropna(inplace=True)
    word_ratio = word_ratio[~np.isinf(word_ratio)]
    # convert to log b/c YOLO
    word_ratio = np.log(word_ratio)
    word_ratio.sort_values(inplace=True, ascending=False)
    return (group_val_1, group_val_2), word_ratio