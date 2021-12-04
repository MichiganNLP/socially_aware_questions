import requests
import pandas as pd
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
from time import sleep
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

from tqdm import tqdm
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