#imports
import pickle, json, multiprocessing, time, random, enchant, nltk, math, os
import pandas as pd
import statsmodels.formula.api as smf
import numpy as np
from collections import Counter
from tokenizer import tokenizer
from nltk.util import ngrams
from gensim.models import Word2Vec
from gensim.test.utils import common_texts
from pathlib import Path
from functools import reduce

#functions: utilities
#
#def GetData(in_path, start, end, offset): 
#    df = pd.DataFrame()
#    cur_start = start
#    
#    while cur_start < end: 
#        cur_end = cur_start + offset
#        print(cur_start, cur_end)
#        cur_start = cur_end
#        for filename in Path(in_path).glob('**/*.csv'):
#            print(filename)
#            #file_split = 
#            cur_df = pd.read_csv(filename, warn_bad_lines=True)
#            df = df.append(cur_df)
#            
#    return df


def GetData(in_path, start, end, offset): 
    df = pd.DataFrame()
    cur_start = start
    
    while cur_start < end: 
        cur_end = cur_start + offset
        print(cur_start, cur_end)
        for directory in Path(in_path).glob('*'):
            if not '.DS_Store' in str(directory): 
                subreddit = str(directory).strip(in_path)
                #special case for some reason, fix later
                if subreddit == 'pple_': 
                    subreddit = 'apple_'
                    
                cur_file = str(directory) + '/' + subreddit + str(cur_start) + '_' + str(cur_end) + '.csv'
                cur_df = pd.read_csv(cur_file, error_bad_lines=False, warn_bad_lines=True, engine='python')
                df = df.append(cur_df)
                
        cur_start = cur_end
        
    return df


def CleanData(in_df, pickled=False): 
    if pickled: 
        cleaned_df = pd.read_pickle('cleaned_df.pkl')
    
    else: 
        #drop nans from the body or time category since they can't be used for later analysis
        cleaned_df = in_df.copy()
        cleaned_df['created_utc'] = pd.to_numeric(cleaned_df['created_utc'], errors='coerce', downcast='integer')
        cleaned_df = cleaned_df.dropna(subset=['body', 'created_utc'])
        
        #remove bots
        bots_path = 'bots.txt'
        with open(bots_path) as file: 
            bots = set(file.read().split())
    
        cleaned_df['bot'] = cleaned_df['author'].map(lambda x: x in bots)
        print("Before bots removal:", cleaned_df.shape)
        cleaned_df = cleaned_df[cleaned_df['bot'] == False]
        print("After bots removal:", cleaned_df.shape)
    
        #remove posts by known spammers
        spammers_path = 'spammers.txt'
        with open(spammers_path) as file: 
            spammers = set(file.read().split())
       
        print("Before spammers removal:", cleaned_df.shape)
        cleaned_df['spammer'] = cleaned_df['author'].map(lambda x: x in spammers)
        cleaned_df = cleaned_df[cleaned_df['spammer'] == False]
        print("After spammers removal:", cleaned_df.shape)
       
        #remove duplicates
        print("Before duplicates removal:", cleaned_df.shape)
        cleaned_df = cleaned_df.drop_duplicates(subset='body')
        print("After duplicates removal:", cleaned_df.shape)
       
        #tokenize
        redditizer = tokenizer.RedditTokenizer(preserve_handles=False, preserve_case=False, preserve_url=False)
        cleaned_df['body'] = cleaned_df['body'].map(lambda x: str(x))
        cleaned_df['tokenized'] = cleaned_df['body'].map(lambda x: redditizer.tokenize(x))
    
        #put back together into one string
        cleaned_df['body'] = cleaned_df['tokenized'].map(lambda x: ' '.join(x))
        
        #lemmatize or stem later
        cleaned_df.to_pickle('cleaned_df.pkl')
    return cleaned_df


def TimeSubset(start, end, df): 
    
    
    subset = df[(df['created_utc'] >= start) & (df['created_utc'] <= end)].copy()
    return subset

#updated version
def makeVocab(sub_df, cutoff=0): 
    texts = sub_df['tokenized'].tolist()
    vocab = Counter()
    for text in texts: 
        vocab.update(text)
    
    if cutoff > 0: 
        trimmed_vocab = Counter(el for el in vocab.elements() if vocab[el] > cutoff)
        return trimmed_vocab
    
    return vocab



#functions: dissemination measures
def getD_U(word, texts, vocab): 
    #calculate numerator (U_w)
    num = 0
    
    user_posts = []
    
    #get the user sub and append to the list in one pass
    for user in texts['author'].unique():
        user_list = texts[texts['author'] == user]['body'].tolist()
        user_sub = ' '.join(user_list).split()
        user_posts.append(user_sub)
        user_set = set(' '.join(user_sub).split())
        if word in user_set: 
            num += 1
            #print(num)
    
    #calculate f_w
    try: 
        f_w = vocab[word]/sum(vocab.values())
    except ZeroDivisionError as zdiv: 
        return np.nan
    
    #calculate e_w
    denom = 0
    
    #instead of iterating over posts, iterate over all texts by  given user
    for combined_posts in user_posts:
        denom += (1 - np.exp(-f_w*len(combined_posts)))
    
    return num/denom

#test for faster calculation
def newD_U(word, user_sets, vocab, user_posts, check=False): 
    if word not in vocab.keys(): 
        return np.nan
    
    #calculate numerator (U_w)
    num = 0

    for word_set in user_sets: 
        if word in word_set:
            num += 1
        
    #calculate f_w
    try: 
        f_w = vocab[word]/sum(vocab.values())
    except ZeroDivisionError as zdiv: 
        return np.nan
    
    #calculate e_w
    denom = 0
    
    #instead of iterating over posts, iterate over all texts by  given user
    for combined_posts in user_posts:
        denom += (1 - np.exp(-f_w*len(combined_posts)))
    
    return num/denom

#test for faster calculation
def newD_UMultiprocess(word, user_sets, vocab, user_posts): 
    #calculate numerator (U_w)
    num = 0

    for word_set in user_sets: 
        if word in word_set:
            num += 1
        
    #calculate f_w
    try: 
        f_w = vocab[word]/sum(vocab.values())
    except ZeroDivisionError as zdiv: 
        return (word, np.nan)
    
    #calculate e_w
    denom = 0
    
    #instead of iterating over posts, iterate over all texts by  given user
    for combined_posts in user_posts:
        denom += (1 - np.exp(-f_w*len(combined_posts)))
    
    return (word, num/denom)

#essentially the same calculation as with D_U, except that you check posts that have the same parent id
def getD_T(word, texts, vocab): 
    #calculate numerator (U_w)
    num = 0
    
    thread_posts = []
    
    #get the thread sub and append to the list in one pass
    for thread in texts['parent_id'].unique():
        thread_list = texts[texts['parent_id'] == thread]['body'].tolist()
        thread_sub = ' '.join(thread_list).split()
        thread_posts.append(thread_sub)
        thread_set = set(' '.join(thread_sub).split())
        if word in thread_set: 
            num += 1
    
    #calculate f_w
    try: 
        f_w = vocab[word]/sum(vocab.values())
    except ZeroDivisionError as zdiv: 
        return np.nan
    
    #calculate e_w
    denom = 0
    
    #instead of iterating over posts, iterate over all texts by  given thread
    for combined_posts in thread_posts:
        denom += (1 - np.exp(-f_w*len(combined_posts)))
    
    return num/denom

#test for faster calculation
def newD_T(word, thread_sets, vocab, thread_posts): 
    if word not in vocab.keys(): 
        return np.nan
    
    #calculate numerator (U_w)
    num = 0
#    num2 = 0
    
    
    #get the thread sub and append to the list in one pass
    for word_set in thread_sets: 
        if word in word_set:
            num += 1
                   
    #calculate f_w
    try: 
        f_w = vocab[word]/sum(vocab.values())
    except ZeroDivisionError as zdiv: 
        return np.nan
    
    #calculate e_w
    denom = 0
    
    #instead of iterating over posts, iterate over all texts by  given thread
    for combined_posts in thread_posts:
        denom += (1 - np.exp(-f_w*len(combined_posts)))
    
    return num/denom

#essentially the same calculation as with D_U, except that you check posts that have the same parent id
def getD_TMultiprocess(word, texts, vocab): 
    #calculate numerator (U_w)
    num = 0
    
    thread_posts = []
    
    #get the thread sub and append to the list in one pass
    for thread in texts['parent_id'].unique():
        thread_list = texts[texts['parent_id'] == thread]['body'].tolist()
        thread_sub = ' '.join(thread_list).split()
        thread_posts.append(thread_sub)
        thread_set = set(' '.join(thread_sub).split())
        if word in thread_set: 
            num += 1
    
    #calculate f_w
    try: 
        f_w = vocab[word]/sum(vocab.values())
    except ZeroDivisionError as zdiv: 
        return (word, np.nan)
    
    #calculate e_w
    denom = 0
    
    #instead of iterating over posts, iterate over all texts by  given thread
    for combined_posts in thread_posts:
        denom += (1 - np.exp(-f_w*len(combined_posts)))
    
    return (word, num/denom)

#calculate trigrams from vocabulary of posts
def getD_L(texts, n): 
    #join all texts together
    joined = ' '.join(texts['body'].tolist())
    
    tokens = joined.split()
    n_grams = list(ngrams(tokens, n))    
    vocab = Counter(tokens)
    
    #more efficient way of generating ngrams in which word appears than 
    # using a map/lambda
    gram_count = dict.fromkeys(vocab.keys(), 0)
    grams = {k: set() for k in vocab.keys()}
    
    for n_gram in n_grams: 
        for word in n_gram: 
            grams[word].add(n_gram)
            gram_count[word] = len(grams[word])
        
    df = pd.DataFrame()
    df['word'] = gram_count.keys()
    df['n_grams'] = gram_count.values()
    df['freq'] = df['word'].map(lambda x: vocab[x])
    df['log_freq'] = df['freq'].map(lambda x: np.log(x))
    df['log_n_grams'] = df['n_grams'].map(lambda x: np.log(x))
    
    #use statsmodels to calculate linear regression, using smf so you don't have to add a constant
    reg = smf.ols('np.log(n_grams) ~ np.log(freq)', data=df).fit()
    
    #predict values for each word: 
    predictions = reg.predict(df['freq'])
    df['pred'] = predictions
    df['D_L'] = df.apply(lambda x: x['log_n_grams'] - x['pred'], axis=1)

    return df

#generate cosine distances matrix
def getD_SCBOW(texts):
    #hyperparameters from Abdul's work, fiddle with them a bit
    #uncomment to retrain model
    try: 
        model = Word2Vec(texts, size=100, window=5, min_count=100, workers=12)
        model.save("word2vec.model")
        # model = Word2Vec.load("word2vec.model")

        vocab = list(model.wv.vocab)
        n = len(vocab)
        print("W2V vocabulary size", n)
        print("Top 10 vocab:", vocab[:10])

        #generate empty matrix of size n x n, where n = vocab size
        pairs = np.zeros((n, n))

        #iterate over all word pairs, avoiding double computation
        for i, word_row in enumerate(vocab): 
            #fill diagonals
            pairs[i, i] = 1.0
            for j, word_col in enumerate(vocab[i+1:]): 
                k = j+i+1
                sim = model.similarity(word_row, word_col)
                pairs[i, k] = sim
                pairs[k, i] = sim
        print(pairs)

    #if there's nothing in the vocabulary that meets min_count, return empty lists
    except RuntimeError as r: 
        print(r)
        pairs = [[]]
        vocab = []
    return (pairs, vocab)

#calculate semantic neighborhood density for an individual word
def SemDensity(word, index, similarities, threshold=None): 
    #if there's a threshold, return integer number of neighbors within that threshold
    if threshold != None: 
        #get column (distances[index]), and check how many words are >= the threshold
        #divide by vocabulary size to normalize
        return len([x for x in similarities[index] if x >= threshold])/len(similarities[index])
    
    #if no threshold, calculate mean distance to all other words in vocabulary
    else: 
        return similarities[index].mean()

def D_SemSeries(df, offset, words, nhood_size): 
    start = df['created_utc'].min()
    end = df['created_utc'].max()
    cur_start = start

    my_dict = dict()
    for word in words: 
        my_dict[word] = []
    
    while cur_start < end: 
        cur_end = cur_start + offset
        print(cur_start, cur_end)

        #subset df, make vocab
        cur_df = TimeSubset(cur_start, cur_end, df) 

        #get SCBOW embeddings if the vocab has size > 0: 
        embed_matrix, embed_vocab = getD_SCBOW(cur_df['tokenized'].tolist())

        if len(embed_vocab) > 0: 
            for word in words: 
                #get its index within the SCBOW vocabulary and then call 
                try: 
                    index = embed_vocab.index(word)
                    D_S = SemDensity(word, index, embed_matrix, nhood_size)

                #if word isn't in vocab: 
                except ValueError: 
                    D_S = np.nan
    
    
                my_dict[word].append((cur_start, D_S))
        else: 
            for word in words: 
                my_dict[word].append((cur_start, np.nan))
        cur_start = cur_end 
        
    return my_dict



#functions: dissemination measure series
def FasterD_USeries(df, offset, words, threshold): 
    start = df['created_utc'].min()
    end = df['created_utc'].max()
    cur_start = start
    
    my_dict = dict()
    for word in words: 
        my_dict[word] = dict()
    
    while cur_start < end: 
        cur_end = cur_start + offset
        print(cur_start, cur_end)

        #subset df, make vocab
        cur_df = TimeSubset(cur_start, cur_end, df) 
        cur_vocab = makeVocab(cur_df, threshold)
        
        #get D_U
        for word in words: 
            D_U = getD_U(word, cur_df, cur_vocab)
            my_dict[word][cur_start] = D_U
        cur_start = cur_end 
    print("my_dict single process: ", my_dict)
    return my_dict


def D_USeries(df, offset, words, threshold): 
    start = df['created_utc'].min()
    end = df['created_utc'].max()
    cur_start = start
    
    my_dict = dict()
    for word in words: 
        my_dict[word] = []
    
    while cur_start < end: 
        cur_end = cur_start + offset
        print(cur_start, cur_end)

        #subset df, make vocab
        cur_df = TimeSubset(cur_start, cur_end, df) 
        cur_vocab = makeVocab(cur_df, threshold)
        
        user_sets = []
        posts = []
        
        #make set of unique threads: 
        for user in cur_df['author'].unique():
            user_list = cur_df[cur_df['author'] == user]['body'].tolist()
            user_sub = ' '.join(user_list).split()
            posts.append(user_sub)
            user_set = set(' '.join(user_sub).split())
            user_sets.append(user_set)
            

        #get D_U
        for word in words: 
            D_U = newD_U(word, user_sets, cur_vocab, posts)
            my_dict[word].append((cur_start, D_U))
        cur_start = cur_end 
        
    return my_dict


def NewD_USeriesMultiprocess(df, offset, words, threshold): 
    start = df['created_utc'].min()
    end = df['created_utc'].max()
    cur_start = start
    
    my_dict = dict()
    for word in words: 
        my_dict[word] = dict()
    
    while cur_start < end: 
        cur_end = cur_start + offset
        print(cur_start, cur_end)

        #subset df, make vocab
        cur_df = TimeSubset(cur_start, cur_end, df) 
        cur_vocab = makeVocab(cur_df, threshold)
        
        user_sets = []
        posts = []
        
        #make set of unique threads: 
        for user in cur_df['author'].unique():
            user_list = cur_df[cur_df['author'] == user]['body'].tolist()
            user_sub = ' '.join(user_list).split()
            posts.append(user_sub)
            user_set = set(' '.join(user_sub).split())
            user_sets.append(user_set)
            

#        #get D_U
#        for word in words: 
#            D_U = newD_U(word, user_sets, cur_vocab, posts)
#            my_dict[word][cur_start] = D_U
            
        #get D_U
        partitions = multiprocessing.cpu_count()
        
        tuples_list = [(word, user_sets, cur_vocab, posts) for word in words]
        with multiprocessing.Pool(partitions) as pool: 
            #return a list of words and d_u values
            results = pool.starmap(newD_UMultiprocess, tuples_list)

        #reconstruct dictionary: 
        for word_tuple in results: 
            my_dict[word_tuple[0]][cur_start] = word_tuple[1]
        cur_start = cur_end 
        
    return my_dict


def FasterD_TSeries(df, offset, words, threshold): 
    start = df['created_utc'].min()
    end = df['created_utc'].max()
    cur_start = start
    
    my_dict = dict()
    for word in words: 
        my_dict[word] = dict()
    
    while cur_start < end: 
        cur_end = cur_start + offset
        print(cur_start, cur_end)

        #subset df, make vocab
        cur_df = TimeSubset(cur_start, cur_end, df) 
        cur_vocab = makeVocab(cur_df, threshold)
        
        #get D_T
        for word in words: 
            D_T = getD_T(word, cur_df, cur_vocab)
            my_dict[word][cur_start] = D_T
        cur_start = cur_end 
    return my_dict


def MultiprocessD_TSeries(df, offset, words, threshold): 
    start = df['created_utc'].min()
    end = df['created_utc'].max()
    cur_start = start
    
    my_dict = dict()
    for word in words: 
        my_dict[word] = dict()
    
    while cur_start < end: 
        cur_end = cur_start + offset
        print(cur_start, cur_end)

        #subset df, make vocab
        cur_df = TimeSubset(cur_start, cur_end, df) 
        cur_vocab = makeVocab(cur_df, threshold)
        
        #get D_T
        partitions = multiprocessing.cpu_count()
        tuples_list = [(word, cur_df, cur_vocab) for word in words]
        with multiprocessing.Pool(partitions) as pool: 
            #return a list of words and d_u values
            results = pool.starmap(getD_TMultiprocess, tuples_list)

        #reconstruct dictionary: 
        for word_tuple in results: 
            my_dict[word_tuple[0]][cur_start] = word_tuple[1]
            
        cur_start = cur_end
    return my_dict


def D_TSeries(df, offset, words, threshold): 
    start = df['created_utc'].min()
    end = df['created_utc'].max()
    cur_start = start
    
    my_dict = dict()
    for word in words: 
        my_dict[word] = []
    
    while cur_start < end: 
        cur_end = cur_start + offset
        print(cur_start, cur_end)

        #subset df, make vocab
        cur_df = TimeSubset(cur_start, cur_end, df) 
        cur_vocab = makeVocab(cur_df, threshold)
        
        thread_sets = []
        posts = []
        
        #make set of unique threads: 
        for thread in cur_df['parent_id'].unique():
            thread_list = cur_df[cur_df['parent_id'] == thread]['body'].tolist()
            thread_sub = ' '.join(thread_list).split()
            posts.append(thread_sub)
            thread_set = set(' '.join(thread_sub).split())
            thread_sets.append(thread_set)

        #get D_T
        for word in words: 
            D_T = newD_T(word, thread_sets, cur_vocab, posts)
            my_dict[word].append((cur_start, D_T))
        cur_start = cur_end 
 
    return my_dict


#get D_L
def D_LSeries(df, offset, words, threshold): 
    start = df['created_utc'].min()
    end = df['created_utc'].max()
    cur_start = start

    my_dict = dict()
    for word in words: 
        my_dict[word] = []
    
    while cur_start < end: 
        cur_end = cur_start + offset

        #subset df, make vocab
        cur_df = TimeSubset(cur_start, cur_end, df) 
        if cur_df.shape[0] <= 0: 
            d_L = np.nan
            #return empty df
            print("EMPTY")
            return pd.DataFrame(columns=['word', 'time', 'd_l'])
        
        else: 
            cur_vocab = makeVocab(cur_df, threshold)
            
            dl_df = getD_L(cur_df, 3)
            for word in words:
                if word in cur_vocab.keys(): 
                    #a bit of a cheap hack, works under assumption that words in df are unique
                    #if time permits: index by word
                    d_L = dl_df[dl_df['word'] == word].iloc[0]['D_L']
                    my_dict[word].append((cur_start, d_L))

                
                else: 
                    my_dict[word].append((cur_start, np.nan))
        
        cur_start = cur_end
               
    return my_dict


def Freq(df, offset, words, threshold): 
    start = df['created_utc'].min()
    end = df['created_utc'].max()
    cur_start = start
    
    my_dict = dict()
    for word in words: 
        my_dict[word] = []
    
    while cur_start < end: 
        cur_end = cur_start + offset

        #subset df, make vocab
        cur_df = TimeSubset(cur_start, cur_end, df) 
        cur_vocab = makeVocab(cur_df, threshold)
        
        for word in words: 
            freq = cur_vocab[word]
            
            #np.nan behavior desirable for computing correlations later
            if freq <= 0: 
                my_dict[word].append((cur_start, np.nan))
            else: 
                my_dict[word].append((cur_start, freq))
        cur_start = cur_end
    
    return my_dict


def RelFreq(df, offset, words, threshold): 
    start = df['created_utc'].min()
    end = df['created_utc'].max()
    cur_start = start
    
    my_dict = dict()
    for word in words: 
        my_dict[word] = []
    
    while cur_start < end: 
        cur_end = cur_start + offset

        #subset df, make vocab
        cur_df = TimeSubset(cur_start, cur_end, df) 
        cur_vocab = makeVocab(cur_df, threshold)
        
        n_words = sum(cur_vocab.values())
        
        for word in words: 
            freq = cur_vocab[word]
            
            #np.nan behavior desirable for computing correlations later
            if freq <= 0: 
                my_dict[word].append((cur_start, np.nan))
            else: 
                my_dict[word].append((cur_start, freq/n_words))
        cur_start = cur_end
    
    return my_dict

 
def Rank(df, offset, words, threshold): 
    start = df['created_utc'].min()
    end = df['created_utc'].max()
    cur_start = start
    
    my_dict = dict()
    for word in words: 
        my_dict[word] = []
    
    while cur_start < end: 
        cur_end = cur_start + offset

        #subset df, make vocab
        cur_df = TimeSubset(cur_start, cur_end, df) 
        cur_vocab = makeVocab(cur_df, threshold)
        
        #generate rank dictionary
        rank_dict = dict((item[1][0], item[0]) for item in enumerate(cur_vocab.most_common()))
        
        for word in words: 
            freq = cur_vocab[word]
            
            #np.nan behavior desirable for computing correlations later
            if freq <= 0: 
                my_dict[word].append((cur_start, np.nan))
            else: 
                my_dict[word].append((cur_start, rank_dict[word]))
        cur_start = cur_end
    
    return my_dict


def NormedRank(df, offset, words, threshold): 
    start = df['created_utc'].min()
    end = df['created_utc'].max()
    cur_start = start
    
    my_dict = dict()
    for word in words: 
        my_dict[word] = []
    
    while cur_start < end: 
        cur_end = cur_start + offset

        #subset df, make vocab
        cur_df = TimeSubset(cur_start, cur_end, df) 
        cur_vocab = makeVocab(cur_df, threshold)
        vocab_size = len(cur_vocab.items())
        
        #generate rank dictionary
        rank_dict = dict((item[1][0], item[0]) for item in enumerate(cur_vocab.most_common()))
        
        for word in words: 
            freq = cur_vocab[word]
            
            #np.nan behavior desirable for computing correlations later
            if freq <= 0: 
                my_dict[word].append((cur_start, np.nan))
            else: 
                my_dict[word].append((cur_start, rank_dict[word]/vocab_size))
        cur_start = cur_end
    
    return my_dict

#monster function that does everything
def CalcMeasures(reddit_df, data_df, words, threshold, measurements): 
    #check that there are posts in the current subset
    if len(reddit_df.index) <= 0: 
        return data_df
    
    cur_vocab = makeVocab(reddit_df, threshold)
    n_words = sum(cur_vocab.values())
    ranks = dict((item[1][0], item[0]) for item in enumerate(cur_vocab.most_common()))
    dl_df = getD_L(cur_df, 3)
    
    #initialize dictionaries
    freq_dict = dict()
    rel_freq_dict = dict()
    rank_dict = dict()
    normed_rank_dict = dict()
    d_u_dict = dict()
    d_t_dict = dict()
    d_l_dict = dict()
    d_s_25_dict = dict()
    d_s_50_dict = dict()
    d_s_75_dict = dict()
    d_s_mean_dict = dict()
    
    for word in words: 
        freq_dict[word] = []
        rel_freq_dict[word] = []
        rank_dict[word] = []
        normed_rank_dict[word] = []
        d_u_dict[word] = []
        d_t_dict[word] = [] 
        d_l_dict[word] = []
        d_s_25_dict[word] = []
        d_s_50_dict[word] = []
        d_s_75_dict[word] = []
        d_s_mean_dict[word] = []
        
    user_sets = []
    user_posts = []
    
    #make set of unique threads: 
    for user in cur_df['author'].unique():
        user_list = cur_df[cur_df['author'] == user]['body'].tolist()
        user_sub = ' '.join(user_list).split()
        user_posts.append(user_sub)
        user_set = set(' '.join(user_sub).split())
        user_sets.append(user_set)
    
    #D^T setup: 
    thread_sets = []
    thread_posts = []
    
    #make set of unique threads: 
    for thread in cur_df['parent_id'].unique():
        thread_list = cur_df[cur_df['parent_id'] == thread]['body'].tolist()
        thread_sub = ' '.join(thread_list).split()
        thread_posts.append(thread_sub)
        thread_set = set(' '.join(thread_sub).split())
        thread_sets.append(thread_set)
    
     
    #get SCBOW embeddings if the vocab has size > 0: 
    embed_matrix, embed_vocab = getD_SCBOW(cur_df['tokenized'].tolist())

    for word in words: 
        freq = cur_vocab[word]
        
        if freq < threshold: 
            #return nan for all other measurements
            freq_dict[word].append((cur_start, np.nan))
            rel_freq_dict[word].append((cur_start, np.nan))
            rank_dict[word].append((cur_start, np.nan))
            normed_rank_dict[word].append((cur_start, np.nan))
            d_u_dict[word].append((cur_start, np.nan))
            d_t_dict[word].append((cur_start, np.nan))
            d_l_dict[word].append((cur_start, np.nan))
            d_s_25_dict[word].append((cur_start, np.nan))
            d_s_50_dict[word].append((cur_start, np.nan))
            d_s_75_dict[word].append((cur_start, np.nan))
            d_s_mean_dict[word].append((cur_start, np.nan))
            
            
        else: 
            #if the frequency is >= threshold, calculate all other measurements
            freq_dict[word].append((cur_start, freq))
            rel_freq_dict[word].append((cur_start, freq/n_words))
            rank_dict[word].append((cur_start, ranks[word]))
            normed_rank_dict[word].append((cur_start, ranks[word]/n_words))
        
            #d_u
            D_U = newD_U(word, user_sets, cur_vocab, user_posts)
            d_u_dict[word].append((cur_start, D_U))
            
            #c_t
            D_T = newD_T(word, thread_sets, cur_vocab, thread_posts)
            d_t_dict[word].append((cur_start, D_T))
        
            #d_l calculation: 
            if word in cur_vocab.keys(): 
                #a bit of a cheap hack, works under assumption that words in df are unique
                #if time permits: index by word
                D_L = dl_df[dl_df['word'] == word].iloc[0]['D_L']
                d_l_dict[word].append((cur_start, D_L))
            
            #an extra check, just to be safe
            else: 
                d_l_dict[word].append((cur_start, np.nan))

            #calculate d_s
            try: 
                index = embed_vocab.index(word)
                d_s_25 = SemDensity(word, index, embed_matrix, 0.25)
                d_s_50 = SemDensity(word, index, embed_matrix, 0.50)
                d_s_75 = SemDensity(word, index, embed_matrix, 0.75)
                d_s_mean = SemDensity(word, index, embed_matrix)


            #if word isn't in vocab: 
            except ValueError: 
                d_s_25 = np.nan
                d_s_50 = np.nan
                d_s_75 = np.nan
                d_s_mean = np.nan
                
            d_s_25_dict[word].append((cur_start, d_s_25))
            d_s_50_dict[word].append((cur_start, d_s_50))
            d_s_75_dict[word].append((cur_start, d_s_75))
            d_s_mean_dict[word].append((cur_start, d_s_mean))


    data_df['freq'] = data_df.apply(lambda x: x['freq']+ freq_dict[x['word']], axis=1)
    data_df['rel_freq'] = data_df.apply(lambda x: x['rel_freq']+ rel_freq_dict[x['word']], axis=1)
    data_df['rank'] = data_df.apply(lambda x: x['rank']+ rank_dict[x['word']], axis=1)
    data_df['normed_rank'] = data_df.apply(lambda x: x['normed_rank']+ normed_rank_dict[x['word']], axis=1)
    data_df['d_u'] = data_df.apply(lambda x: x['d_u']+ d_u_dict[x['word']], axis=1)
    data_df['d_t'] = data_df.apply(lambda x: x['d_t']+ d_t_dict[x['word']], axis=1)
    data_df['d_l'] = data_df.apply(lambda x: x['d_l']+ d_l_dict[x['word']], axis=1)  
    data_df['d_s_25'] = data_df.apply(lambda x: x['d_s_25']+ d_s_25_dict[x['word']], axis=1) 
    data_df['d_s_50'] = data_df.apply(lambda x: x['d_s_50']+ d_s_50_dict[x['word']], axis=1) 
    data_df['d_s_75'] = data_df.apply(lambda x: x['d_s_75']+ d_s_75_dict[x['word']], axis=1) 
    data_df['d_s_mean'] = data_df.apply(lambda x: x['d_s_mean']+ d_s_mean_dict[x['word']], axis=1) 
    
    print(data_df.head(20))
    return data_df



if __name__ == "__main__": 
    #for timing: 
    start_time = time.time()
    
    #create words + measurements df, intialized to empty lists
    data_df = pd.read_csv('words.csv')  
    my_measurements = ['freq', 'rel_freq', 'rank', 'normed_rank', 'd_u', 'd_t', 'd_l', 'd_s_25', 'd_s_50', 'd_s_75', 'd_s_mean']
    for measurement in my_measurements: 
        data_df[measurement] = data_df['word'].map(lambda x: [])
    print(data_df.head())
    words = data_df['word'].tolist()
    
    #unix timestamp for earliest df
    cur_start = 1451606400
    offset = 2629743
    end = 1485793059
    threshold = 5
  
    
    while cur_start < end: 
        cur_end = cur_start + offset
        cur_df =  CleanData(GetData('data/', cur_start, cur_end, offset))
        data_df = CalcMeasures(cur_df, data_df, words, 5, my_measurements)
        
        cur_start = cur_end
    
    #filter out words that never hit the frequency threshold
    data_df['no_values'] = data_df['freq'].map(lambda x: all(np.isnan(y[1]) for y in x))
    print(data_df['no_values'].value_counts())
    
    data_df = data_df[data_df['no_values'] == False]
    
    data_df.to_csv('data_df.csv')
    data_df.to_pickle('data_df.pkl')
    
    end_time = time.time()
    print("Total time elapsed:", str(end_time-start_time))