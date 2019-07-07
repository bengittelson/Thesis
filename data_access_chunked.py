#imports
import pandas as pd
from psaw import PushshiftAPI
import time, os

def GetSubreddit(subr, in_api, start=None, end=None, offset=None):
    #create folder for subreddit
    folder_name = 'data/' + subr + '_data'
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)

    #iterate over time intervals and call search_comments
    cur_start = start

    fields = ['author', 'author_fullname', 'body', 'created', 'created_utc', 'edited', 'id', 'is_submitter', 'link_id', 'parent_id', 'permalink', 'retrieved_on', 'stickied', 'subreddit', 'subreddit_id', 'updated_utc']

    while cur_start < end:
        cur_end = cur_start + offset
        print(cur_start, cur_end)

        #do everything below and output to csv
        #add limit = ... to limit to 100, 1000, etc. posts from each interval
        gen = in_api.search_comments(subreddit=subr, after=cur_start, before=cur_end, filter=fields, limit=100)
        cache = []

        print('\t got gen')
        for c in gen:
            cache.append(c)

        # If you really want to: pick up where we left off to get the rest of the results.
        if False:
            print('.', end='')
            for c in gen:
                cache.append(c)

        comments = [post.d_ for post in cache]
        print('\t made comments')

        df = pd.DataFrame(comments)
        print('\t made df')
        print(df.memory_usage(deep=True).sum())
        df.to_csv(folder_name + '/' + subr + '_' + str(cur_start) + '_' + str(cur_end) + '.csv')
        cur_start = cur_end

if __name__ == "__main__":
    subr_names = ['Android', 'hockey', 'FinalFantasy', 'subaru', 'photography', 'baseball', 'beer', 'boardgames', 'cars', 'Guitar', 'harrypotter', 'Patriots', 'pcgaming', 'pokemon', 'poker', 'reddevils', 'running', 'StarWars', 'apple', 'Liverpool']

    api = PushshiftAPI()
    if not os.path.isdir('data'): 
        os.mkdir('data')

    for name in subr_names:
        print(name)

        #current parameters: 2005-01-01 00:00:00 to 2020-01-01 00:00:00, at ~1 month intervals (30.44 days)
        #switched to 2017-2018 at ~1 month intervals
        GetSubreddit(name, api, 1451606400, 1483228800, 2629743)

