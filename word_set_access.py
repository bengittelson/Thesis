import pandas as pd
import enchant
from nltk.corpus import names
import wikipedia

#helper function for urbandictionary lambda function
def CheckWikipedia(title): 
    try: 
        wikipedia.page(title)
        return True
    except wikipedia.exceptions.PageError: 
        return False
    except wikipedia.exceptions.DisambiguationError: 
        return True

#to access urban dictionary data, returns a list of nonstandard words
def UrbanAccess(file_path, single_word=True, pickled=False): 
    if pickled: 
        nonstandard = pd.read_pickle('urban.pkl')
    else: 
        urban = pd.read_csv(file_path)
        
        us = enchant.Dict("en_US")
        uk = enchant.Dict("en_UK")
        gb = enchant.Dict("en_UK")
        ca = enchant.Dict("en_CA")
        au = enchant.Dict("en_AU")
        ind = enchant.Dict("en_IN")
        urban['enchant'] = urban['word'].map(lambda x: us.check(str(x)) or uk.check(str(x)) or gb.check(str(x)) or ca.check(str(x)) or au.check(str(x)) or ind.check(str(x)))
        
        #clear names
        urban['name'] = urban['word'].map(lambda x: x in set(names.words()))
        print(urban['name'].value_counts())
        
        #drop duplicates
        urban = urban.drop_duplicates(subset='word')
        
        nonstandard = urban[(urban['enchant'] == False) & (urban['name'] == False)]
        print(nonstandard['name'].value_counts())
        nonstandard['multiword'] = nonstandard['word'].map(lambda x: ' ' in str(x))
    
    if single_word: 
        single_word = nonstandard[nonstandard['multiword'] == False]
        print("Single word before:", single_word.shape)

        #make wikipedia call, positined here to limit number of calls
        single_word['wikipedia'] = single_word['word'].map(lambda x: CheckWikipedia(x))
        single_word = single_word[single_word['wikipedia'] == True]
        print("Single word after:", single_word.shape)
        nonstandard.to_pickle('urban.pkl')
        return single_word['word'].tolist()
        
    else: 
        multi_word = nonstandard[nonstandard['multiword'] == True]
        #make wikipedia call, positined here to limit number of calls
        single_word['wikipedia'] = single_word['word'].map(lambda x: CheckWikipedia(x))
        nonstandard.to_pickle('urban.pkl')
        return multi_word['word'].tolist()
    

def NUTSAccess(nuts_files): 
    entity = -1
    label = -1
    entities = set()
    
    for nuts_file in nuts_files: 
        with open(nuts_file, 'r') as file: 
            for line in file: 
                split = line.split()
                if len(split) == 2: 
                    tag = split[1]
                    if tag.startswith('B-'): 
                        entity = split[0]
                        label = tag.strip('B-')
                    if tag.startswith('I-'): 
                        entity = entity + '_' + split[0]
                else: 
                    if entity != -1 and label != -1: 
                        entity_tuple = (entity, label)
                        entities.add(entity_tuple)

    df = pd.DataFrame(list(entities), columns=['word', 'nuts_ent_label'])
    return df


def CoNLLAccess(conll_file): 
    with open(conll_file, 'r') as file: 
        splits = [line.split() for line in file]
        
    splits = [split for split in splits if len(split) >= 2]
    ents = [(x[0], '_'.join(x[1:])) for x in splits]
    
    df = pd.DataFrame(ents, columns=['conll_ent_label', 'word'])
    return df


def BNCAccess(bnc_file): 
    with open(bnc_file, 'r') as file: 
        splits = [tuple(line.split()) for line in file]
    
    bnc_df = pd.DataFrame(splits, columns=['bnc_freq', 'word', 'pos', 'bnc_docs'])
    
    us = enchant.Dict("en_US")
    uk = enchant.Dict("en_UK")
    gb = enchant.Dict("en_UK")
    ca = enchant.Dict("en_CA")
    au = enchant.Dict("en_AU")
    ind = enchant.Dict("en_IN")
    bnc_df['enchant'] = bnc_df['word'].map(lambda x: us.check(str(x)) or uk.check(str(x)) or gb.check(str(x)) or ca.check(str(x)) or au.check(str(x)) or ind.check(str(x)))
    df = bnc_df[bnc_df['enchant'] == True]
    df = df.drop(labels='enchant', axis=1)
    return df

    
if __name__ == '__main__': 
    nuts_paths = ['wnut17train.conll', 'emerging.dev.conll', 'emerging.test.annotated']
    nuts = NUTSAccess(nuts_paths)
    nuts['source'] = 'nuts'
    conll = CoNLLAccess('eng.list')
    conll['source'] = 'conll'
    
    #urban dictionary words
    urban_dict = UrbanAccess('urban_cleaned.csv')
    urban = pd.DataFrame(columns=['word', 'source'])
    urban['word'] = urban_dict
    urban['source'] = 'urban'
    
    #bnc words
    bnc = BNCAccess('written.num.o5')
    
    words = pd.concat([nuts, conll, urban, bnc])
    print(words.head())
    
    for df in [nuts, conll, urban, bnc]: 
        print(df.shape)
    
    