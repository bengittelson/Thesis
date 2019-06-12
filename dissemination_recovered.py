#imports
import pickle, json, multiprocessing
import pandas as pd
import statsmodels.formula.api as smf
import numpy as np
from collections import Counter
from tokenizer import tokenizer
from nltk.util import ngrams
from gensim.models import Word2Vec
from gensim.test.utils import common_texts

#functions: utilities
def CleanData(in_df): 
    # cleaned_df = in_df.copy()
    
    # #remove bots
    # bots_path = '/Users/benjamingittelson/Documents/SDS/Thesis/bots.txt'
    # with open(bots_path) as file: 
    #     bots = set(file.read().split())

    # cleaned_df['bot'] = cleaned_df['author'].map(lambda x: x in bots)

    # print("Before bots removal:", cleaned_df.shape)
    # cleaned_df = cleaned_df[cleaned_df['bot'] == False]
    # print("After bots removal:", cleaned_df.shape)

    # #remove posts by known spammers
    # spammers_path = '/Users/benjamingittelson/Documents/SDS/Thesis/spammers.txt'
    # with open(spammers_path) as file: 
    #     spammers = set(file.read().split())
    
    # print("Before spammers removal:", cleaned_df.shape)
    # cleaned_df['spammer'] = cleaned_df['author'].map(lambda x: x in spammers)
    # cleaned_df = cleaned_df[cleaned_df['spammer'] == False]
    # print("After spammers removal:", cleaned_df.shape)
    
    # #remove duplicates
    # print("Before duplicates removal:", cleaned_df.shape)
    # cleaned_df = cleaned_df.drop_duplicates(subset='body')
    # print("After duplicates removal:", cleaned_df.shape)
    
    # #tokenize
    # redditizer = tokenizer.RedditTokenizer(preserve_handles=False, preserve_case=False, preserve_url=False)
    # cleaned_df['tokenized'] = cleaned_df['body'].map(lambda x: redditizer.tokenize(x))

    # #put back together into one string
    # cleaned_df['body'] = cleaned_df['tokenized'].map(lambda x: ' '.join(x))
    
    # #lemmatize or stem later
    # cleaned_df.to_pickle('cleaned_df.pkl')
    cleaned_df = pd.read_pickle('cleaned_df.pkl')
    return cleaned_df


def TimeSubset(start, end, df): 
    subset = df[(df['created_utc'] >= start) & (df['created_utc'] <= end)].copy()
    return subset

@profile
def makeVocab(sub_df, cutoff=0): 
    texts = ' '.join(sub_df['body'].tolist())
    t_tizer = tokenizer.RedditTokenizer(preserve_handles=False, preserve_case=False, preserve_url=False)
    tokens = t_tizer.tokenize(texts)
    vocab = Counter(tokens)
    
    #filter out words below a frequency cutoff, following Altmann et al. 2011
    if cutoff > 0: 
        trimmed_vocab = Counter(el for el in vocab.elements() if vocab[el] > cutoff)
        return trimmed_vocab
    
    return vocab
@profile
def W2VVocab(sub_df, cutoff=0): 
#    dummy = Word2Vec(iter=1, min_count=cutoff)
#    
#    dummy.build_vocab(sub_df['body'].tolist())
#    
#    #check parameters later
#    voc = dummy.vocabulary.prepare_vocab(hs=0, negative=0, wv= min_count=cutoff)
#    print(voc)
    
    #test with sklearn
    
    #test without
    
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

#same as regular D_U calculation, except that we also return the word 

def getD_UMultiprocess(word, texts, vocab): 
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
    print("N_grams test:", n_grams[:10])
    vocab = Counter(tokens)
    
    #calculate number of n-grams in which each word appears and add as column to pandas df
    df = pd.DataFrame()
    df['word'] = vocab.keys()
    df['freq'] = df['word'].map(lambda x: vocab[x])
    df['log_freq'] = df['freq'].map(lambda x: np.log(x))
    df['n_grams'] = df['word'].map(lambda x: len(set([y for y in n_grams if x in y])))
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

def SemDensity(word, index, distances, threshold=None): 
    #if there's a threshold, return integer number of neighbors within that threshold
    if threshold != None: 
        #get column (distances[index]), and check how many words are >= the threshold
        #divide by vocabulary size to normalize
        return len([x for x in distances[index] if x >= threshold])/len(distances[index])
    
    #if no threshold, calculate mean distance to all other words in vocabulary
    else: 
        return distances[index].mean()


def FasterD_SSeries(df, offset, words, nhood_size): 
    start = df['created_utc'].min()
    end = df['created_utc'].max()
    cur_start = start
    
    #initialize the cumulative vocab as an empty set
    cum_vocab = Counter()
    my_dict = dict()
    for word in words: 
        my_dict[word] = dict()
    
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
    
                my_dict[word][cur_start] = D_S

        else: 
            for word in words: 
                my_dict[word][cur_start] = np.nan
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


def MultiprocessD_USeries(df, offset, words, threshold): 
    start = df['created_utc'].min()
    end = df['created_utc'].max()
    cur_start = start
    
    #initialize the cumulative vocab as an empty set
    cum_vocab = Counter()
    my_dict = dict()
    for word in words: 
        my_dict[word] = dict()
    
    while cur_start < end: 
        cur_end = cur_start + offset
        print(cur_start, cur_end)

        #subset df, make vocab
        cur_df = TimeSubset(cur_start, cur_end, df)
        print("Cur_df shape:", cur_df.shape) 
        cur_vocab = makeVocab(cur_df, threshold)
        
        #get D_U
        partitions = multiprocessing.cpu_count()
        tuples_list = [(word, cur_df, cur_vocab) for word in words]
        with multiprocessing.Pool(partitions) as pool: 
            #return a list of words and d_u values
            results = pool.starmap(getD_UMultiprocess, tuples_list)

        #reconstruct dictionary: 
        for word_tuple in results: 
            print(word_tuple)
            my_dict[word_tuple[0]][cur_start] = word_tuple[1]
            
        cur_start = cur_end

    return my_dict

@profile
def FasterD_TSeries(df, offset, words, threshold): 
    start = df['created_utc'].min()
    end = df['created_utc'].max()
    cur_start = start
    
    #initialize the cumulative vocab as an empty set
    cum_vocab = Counter()
    my_dict = dict()
    for word in words: 
        my_dict[word] = dict()
    
    while cur_start < end: 
        cur_end = cur_start + offset
        print(cur_start, cur_end)

        #subset df, make vocab
        cur_df = TimeSubset(cur_start, cur_end, df) 
        cur_vocab = makeVocab(cur_df, threshold)
        
        #test w2v vocab
        w2v_vocab = W2VVocab(cur_df, threshold)
        
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

#get D_L

def FasterD_LSeries(df, offset, words, threshold): 
    start = df['created_utc'].min()
    end = df['created_utc'].max()
    cur_start = start
    my_dict = dict()
    for word in words: 
        my_dict[word] = dict()
    
    while cur_start < end: 
        cur_end = cur_start + offset

        #subset df, make vocab
        cur_df = TimeSubset(cur_start, cur_end, df) 
        if cur_df.shape[0] <= 0: 
            d_L = np.nan
        
        else: 
            cur_vocab = makeVocab(cur_df, threshold)
            
            dl_df = getD_L(cur_df, 3)
            for word in words:
                if word in cur_vocab.keys(): 
                    #a bit of a cheap hack, works under assumption that words in df are unique
                    #if time permits: index by word
                    d_L = dl_df[dl_df['word'] == word].iloc[0]['D_L']
                    my_dict[word][cur_start] = d_L

                
                else: 
                    d_L = np.nan
                    my_dict[word][cur_start] = d_L
        
        cur_start = cur_end
            
    return my_dict


def FreqFaster(df, offset, words, threshold): 
    start = df['created_utc'].min()
    end = df['created_utc'].max()
    cur_start = start
    
    #initialize the cumulative vocab as an empty set
    cum_vocab = Counter()
    my_dict = dict()
    
    #nested dict structure
    for word in words: 
        my_dict[word] = dict()
    
    while cur_start < end: 
        cur_end = cur_start + offset

        #subset df, make vocab
        cur_df = TimeSubset(cur_start, cur_end, df) 
        cur_vocab = makeVocab(cur_df, threshold)
        
        for word in words: 
            freq = cur_vocab[word]
            
            #np.nan behavior desirable for computing correlations later
            if freq <= 0: 
                my_dict[word][cur_start] = np.nan
            else: 
                my_dict[word][cur_start] = freq
        cur_start = cur_end
    
    return my_dict


def Rank(df, offset, words, threshold): 
    start = df['created_utc'].min()
    end = df['created_utc'].max()
    cur_start = start
    
    my_dict = dict()
    
    #nested dict structure
    for word in words: 
        my_dict[word] = dict()
    
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
                my_dict[word][cur_start] = np.nan
            else: 
                my_dict[word][cur_start] = rank_dict[word]
        cur_start = cur_end
    
    return my_dict


def NormedRank(df, offset, words, threshold): 
    start = df['created_utc'].min()
    end = df['created_utc'].max()
    cur_start = start
    
    #initialize the cumulative vocab as an empty set
    cum_vocab = Counter()
    my_dict = dict()
    
    #nested dict structure
    for word in words: 
        my_dict[word] = dict()
    
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
                my_dict[word][cur_start] = np.nan
            else: 
                
                #replace with rank
                my_dict[word][cur_start] = rank_dict[word]/vocab_size
        cur_start = cur_end
    
    return my_dict

#main
if __name__ == "__main__": 
    #read in data
    reddit_df = pd.read_csv('final_df.csv')
    
    #access warriner, retrieve top 250 by emotional valence
    warriner = pd.read_csv('warriner_affect_ratings.csv')
    top_250 = warriner.nlargest(250, 'V.Mean.Sum')
    bottom_250 = warriner.nsmallest(250, 'V.Mean.Sum')
    test_words = top_250['Word'].tolist() + bottom_250['Word'].tolist()
    
    data_df = pd.DataFrame(columns=['word'])
    
    #cutoff added to test
    data_df['word'] = test_words

    
    cleaned = CleanData(reddit_df)


    test_df = cleaned[cleaned['created_utc'] <= 1551694400]
    print(test_df.shape)

#    multiprocess_d_t = MultiprocessD_TSeries(test_df, 2629743, data_df['word'].tolist(), 5)
    d_t = FasterD_TSeries(test_df, 2629743, data_df['word'].tolist(), 5)
 

    # matrix, voc = getD_SCBOW(cleaned['tokenized'].tolist())

    # for i, word in enumerate(voc): 
    #     print(word + ":", SemDensity(word, i, matrix), SemDensity(word, i, matrix, threshold=0.5))

    #semantic neighborhood density 
#     freq = FreqFaster(cleaned, 2629743, data_df['word'].tolist(), 5)
#     data_df['freq'] = data_df['word'].map(lambda x: freq[x])
#     print("Frequency completed")

    
#     #frequency
#    freq = FreqFaster(cleaned, 2629743, data_df['word'].tolist(), 5)
#    data_df['freq'] = data_df['word'].map(lambda x: freq[x])
#    print("Frequency completed")
#    
#     #rank
#    rank = Rank(cleaned, 2629743, data_df['word'].tolist(), 5)
#    data_df['rank'] = data_df['word'].map(lambda x: rank[x])
#    print("Rank completed")
#    
#     #normed rank
#    normed_rank = NormedRank(cleaned, 2629743, data_df['word'].tolist(), 5)
#    data_df['normed_rank'] = data_df['word'].map(lambda x: normed_rank[x])
#    print("Normed rank completed")

    # #semantic neighborhood density; note that the final parameter here is for the 
    # # similarity cutoff, not the frequency cutoff, which is set above
    # d_s = FasterD_SSeries(cleaned, 2629743, data_df['word'].tolist(), 0.5)
    # data_df['rank'] = data_df['word'].map(lambda x: rank[x])
    # print("Semantic dissemination completed")

    # print(data_df.head())
    # data_df.to_pickle('data_df.pkl')

    


    
#     #d_u
#     d_u = FasterD_USeries(cleaned, 2629743, data_df['word'].tolist(), 5)
#     data_df['d_u'] = data_df['word'].map(lambda x: d_u[x])
#     print("D^U completed")
    
#     #d_t
#     d_t = FasterD_TSeries(cleaned, 2629743, data_df['word'].tolist(), 5)
#     data_df['d_t'] = data_df['word'].map(lambda x: d_t[x])
#     print("D^T completed")

#     data_df.to_csv('data_df_no_d_l.csv')
    
#     #d_l 
#     d_l = FasterD_LSeries(cleaned, 2629743, data_df['word'].tolist(), 5)
#     data_df['d_l'] = data_df['word'].map(lambda x: d_l[x])
#     print("D^L completed")


    

    
    
