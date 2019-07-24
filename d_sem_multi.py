#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 10:41:06 2019

@author: benjamingittelson
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 22:54:23 2019

@author: benjamingittelson
"""
import time, string, multiprocessing, gensim
import pandas as pd
import numpy as np
from gensim.models import FastText
from gensim.test.utils import datapath
from pathlib import Path
from tokenizer import tokenizer
from collections import Counter

#functions: utilities
def GetData(in_path, start, end, data_offset, subreddits=None, random_sample=None): 
    print(start, end)
    df = pd.DataFrame()
    
    if subreddits: 
        for subreddit in subreddits: 
            if subreddit == 'pple_': 
                subreddit = 'apple_'

            data_start = start
            while data_start < end: 
                data_end = data_start + data_offset
                cur_file = in_path + subreddit + '_data' + '/' + subreddit + '_' + str(data_start) + '_' + str(data_end) + '.csv'
                try: 
                    cur_df = pd.read_csv(cur_file, error_bad_lines=False, warn_bad_lines=True, engine='python')
                    df = df.append(cur_df)
                except FileNotFoundError: 
                    print("%s not found" % cur_file)
                data_start = data_end
            
    else: 
        for directory in Path(in_path).glob('*'):
            if not '.DS_Store' in str(directory):  
                subreddit = str(directory).strip(in_path)
                
                
                #special case for some reason, fix later
                if subreddit == 'pple_': 
                    subreddit = 'apple_'
                
                data_start = start
                while data_start < end: 
                    data_end = data_start + data_offset
                    cur_file = str(directory) + '/' + subreddit + str(data_start) + '_' + str(data_end) + '.csv'
                    try: 
                        cur_df = pd.read_csv(cur_file, error_bad_lines=False, warn_bad_lines=True, engine='python')
                        df = df.append(cur_df)
                    except FileNotFoundError: 
                        print("%s not found" % cur_file)
                    data_start = data_end
    
    if random_sample: 
        try:
            return df.sample(n=random_sample)
        except ValueError:
            return df
    else: 
        return df


def CleanData(in_df, pickled=False): 
    if pickled: 
        cleaned_df = pd.read_pickle('cleaned_df.pkl')
    
    else: 
        #drop nans from the body or time category since they can't be used for later analysis
        cleaned_df = in_df
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
        
        #text to lowercase
        cleaned_df['body'] = cleaned_df['body'].map(lambda x: str(x).translate(str.maketrans('', '', string.punctuation)))
        cleaned_df['tokenized'] = cleaned_df['body'].map(lambda x: redditizer.tokenize(x))
    
        #put back together into one string
        cleaned_df['body'] = cleaned_df['tokenized'].map(lambda x: ' '.join(x))
        
        #lemmatize or stem later
        cleaned_df.to_pickle('cleaned_df.pkl')
        print(cleaned_df['tokenized'].head())
    return cleaned_df


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


def D_Sem(word, column, cur_start, threshold=None): 
    if column is None: 
        return (word, cur_start, np.nan)
    else: 
        if threshold != None: 
            return (word, cur_start, len([x for x in column if x >= threshold])/len(column))
        else: 
            return (word, cur_start, column.mean())

def D_SemSeries(sim_cols, start, sim_cutoff, n_partitions):     
    word_input = [(word, col, cur_start, sim_cutoff) for word, col in sim_cols.items()]
    
    with multiprocessing.Pool(n_partitions) as pool: 
        sems_list = pool.starmap(D_Sem, word_input)
    
    #to free up memory
    word_input = None
    sems_dict = {x[0]: (x[1], x[2]) for x in sems_list}
    sems_list = None
    return sems_dict
        
#    for word, col in sim_cols.items(): 
#        my_D_Sem = D_Sem(col, sim_cutoff)
#        print(word, ": ", str(my_D_Sem))

            

#generate cosine distances matrix
def GetSims(texts, my_words, n_partitions):
    try:  
        ft = FastText(size=200, window=5, min_count=5, sg=1, hs=0, negative=20, workers=n_partitions)
        ft.build_vocab(sentences=texts)
        ft.train(sentences=texts, total_examples=len(texts), epochs=10)
        #sanity check: 
        print()
        ft_eval = ft.wv.evaluate_word_pairs(datapath('wordsim353.tsv'))
        print("FT performance\n\tPearson: %s\n\tFT Spearman: %s\n\tFT OOV ratio: %s" % (str(ft_eval[0]), str(ft_eval[1]), str(ft_eval[2])))
        
        vocab = set(ft.wv.vocab)
        n = len(vocab)
        print("FT vocabulary size", n)
        
        similarity_matrix = []
        index = gensim.similarities.MatrixSimilarity(gensim.matutils.Dense2Corpus(ft.wv.syn0.T))

        for sims in index:
            similarity_matrix.append(sims)
        similarity_array = np.array(similarity_matrix)
        print(similarity_array[:5, :5])
        print(similarity_array.shape)
        
        index_dict = {word: i for i, word in enumerate(ft.wv.index2word)}
        
        word_cols = {word: (similarity_array[index_dict[word]] if word in vocab else None) for word in words}
        return word_cols
        

#    #if there's nothing in the vocabulary that meets min_count, return empty lists
    except RuntimeError as r: 
        print(r)
        return {word: None for word in words}

def D_SemHelper(my_sims, my_words, threshold, n_partitions, cur_time, vocab_set): 
#    sem_input = [(word, my_sims, vocab_set, threshold) for word in my_words]
#    
#    with multiprocessing.Pool(n_partitions) as pool: 
#        d_sems = pool.starmap(D_Sem, sem_input)
    d_sems_list = []
    for word in my_words: 
        d_sems_list.append(D_Sem(word, my_sims, vocab_set, threshold))
    
    d_sems_dict = {item[0]: (item[1], item[2]) for item in d_sems_list}
    return d_sems_dict
    

if __name__ == '__main__': 
    origin = time.time()
    #create words + measurements df, intialized to empty lists
    data_df = pd.read_csv('words.csv')  
    my_measurements = ['d_s_85', 'd_s_75', 'd_s_95', 'd_s_mean']
    for measurement in my_measurements: 
        data_df[measurement] = data_df['word'].map(lambda x: [])
    words = data_df['word'].tolist()
    
    #unix timestamp for earliest df
    cur_start = 1451606400
    data_interval = 2629743
    analysis_interval = 2629743 * 6
    end = 1485793059
    cutoff = 5
    subr_list = ['apple', 'subaru', 'harrypotter', 'Liverpool']
    partitions = 2 * int(multiprocessing.cpu_count()/3)
  
    
    while cur_start < end: 
        #for timing: 
        start_time = time.time()
        cur_end = cur_start + analysis_interval
        raw_df =  GetData('data/', cur_start, cur_end, data_interval, subr_list)
        if raw_df.shape[1] > 0:
            cur_df = CleanData(raw_df) 
            trimmed_vocab = makeVocab(cur_df, cutoff)
            vocab_set = set(trimmed_vocab.keys())
            word_sims = GetSims(cur_df['tokenized'].tolist(), words, partitions)
            
            
            d_s_85 = D_SemSeries(word_sims, cur_start, 0.85, partitions)
            d_s_75 = D_SemSeries(word_sims, cur_start, 0.75, partitions)
            d_s_95 = D_SemSeries(word_sims, cur_start, 0.95, partitions)
            d_s_mean = D_SemSeries(word_sims, cur_start, None, partitions)
            
            data_df['d_s_85'] = data_df.apply(lambda x: x['d_s_85'] + [d_s_85[x['word']]], axis=1)
            data_df['d_s_75'] = data_df.apply(lambda x: x['d_s_75'] + [d_s_75[x['word']]], axis=1)
            data_df['d_s_95'] = data_df.apply(lambda x: x['d_s_95'] + [d_s_95[x['word']]], axis=1)
            data_df['d_s_mean'] = data_df.apply(lambda x: x['d_s_mean'] + [d_s_mean[x['word']]], axis=1)
            
            data_df.to_csv('d_sem_df_5.csv')
            data_df.to_pickle('d_sem_df_5.pkl')
        cur_start = cur_end
        
        end_time = time.time()
        print("Time elapsed for this iteration:", str(end_time-start_time))
        print("Total time elapsed:", str(end_time-origin))
    

