#imports
import pickle, json, multiprocessing, time, random, enchant, nltk, math
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
def GetData(in_path, start, end, offset): 
    df = pd.DataFrame()
    cur_start = start
    
    while cur_start < end: 
        cur_end = cur_start + offset
        print(cur_start, cur_end)
        cur_start = cur_end
        for filename in Path(in_path).glob('**/*.csv'):
            cur_df = pd.read_csv(filename)
            df = df.append(cur_df)
            
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

#main
if __name__ == "__main__": 
    #read in data
    reddit_df = GetData('data/', 0, 1559214151, 1559214151)
    print("DataFrame shape:", reddit_df.shape)
    
    data_df = pd.read_csv('words.csv')    
    cleaned = CleanData(reddit_df, pickled=True)
    print("Cleaned DataFrame shape:", cleaned.shape)    

    #frequency
    freq = Freq(cleaned, 2629743, data_df['word'].tolist(), 5)
    data_df['freq'] = data_df['word'].map(lambda x: freq[x])
    print("Frequency completed")
    
    #filter out words that never hit the frequency threshold
    data_df['no_values'] = data_df['freq'].map(lambda x: all(np.isnan(y[1]) for y in x))
    print(data_df['no_values'].value_counts())
    
    data_df = data_df[data_df['no_values'] == False]
    print(data_df['source'].value_counts())
    
    #relative frequency
    rel_freq = RelFreq(cleaned, 2629743, data_df['word'].tolist(), 5)
    data_df['rel_freq'] = data_df['word'].map(lambda x: rel_freq[x])
    print("Relative frequency completed")
    
    #rank
    rank = Rank(cleaned, 2629743, data_df['word'].tolist(), 5)
    data_df['rank'] = data_df['word'].map(lambda x: rank[x])
    print("Rank completed")
    
    #normed rank
    normed_rank = NormedRank(cleaned, 2629743, data_df['word'].tolist(), 5)
    data_df['normed_rank'] = data_df['word'].map(lambda x: normed_rank[x])
    print("Normed rank completed")
    
    #d_l
    d_l = D_LSeries(cleaned, 2629743, data_df['word'].tolist(), 5)
    data_df['d_l'] = data_df['word'].map(lambda x: d_l[x])
    print("D^L completed")
    
    #d_u
    d_u = D_USeries(cleaned, 2629743, data_df['word'].tolist(), 5)
    data_df['d_u'] = data_df['word'].map(lambda x: d_u[x])
    print("D^U completed")
    
    #d_t
    d_t = D_TSeries(cleaned, 2629743, data_df['word'].tolist(), 5)
    data_df['d_t'] = data_df['word'].map(lambda x: d_t[x])
    print("D^T completed")
    
    #d_s_25
    d_s_25 = D_SemSeries(cleaned, 2629743, data_df['word'].tolist(), 0.25)
    data_df['d_s_25'] = data_df['word'].map(lambda x: d_s_25[x])
    print("D^S 0.25 completed")
    
    #d_s_50
    d_s_50 = D_SemSeries(cleaned, 2629743, data_df['word'].tolist(), 0.50)
    data_df['d_s_50'] = data_df['word'].map(lambda x: d_s_50[x])
    print("D^S 0.50 completed")
    
    #d_s_75
    d_s_75 = D_SemSeries(cleaned, 2629743, data_df['word'].tolist(), 0.75)
    data_df['d_s_75'] = data_df['word'].map(lambda x: d_s_75[x])
    print("D^S 0.75 completed")
    
    #d_s_mean
    d_s_mean = D_SemSeries(cleaned, 2629743, data_df['word'].tolist(), None)
    data_df['d_s_mean'] = data_df['word'].map(lambda x: d_s_mean[x])
    print("D^S mean completed")
    
    #d_s_lsa
    
    
    data_df.to_csv('data_df.csv')