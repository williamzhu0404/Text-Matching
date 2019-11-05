# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 23:55:11 2019
@author: William Zhu
"""

NEGATION_MARKERS = """never nor unless n’t nothing cannot neither no nobody none not
 n‘t nowhere n't""".split()

from gensim import corpora, models, similarities
import numpy as np
import spacy


#spacy_nlp = spacy.load('en')

#Remove stopwords and punctuation
def preprocess(text):
    spacy_nlp = spacy.load('en')
    spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS
    doc = spacy_nlp(text)
    tokens = [token for token in doc if (not token.is_stop) and (not token.is_punct)]
    return tokens


def negation_analysis(text):
    spacy_nlp = spacy.load('en')
    doc = spacy_nlp(text)
    tokens = [token.lemma_ for token in doc if (not token.is_punct)]
    negation = [token.lemma_ for token in doc if token.lemma_ in NEGATION_MARKERS or "n't" in token.lemma_]
    return len(negation) % 2 == 0


def post_tag(tokens):
    pos_tagged = [(token, token.tag_) for token in tokens]
    return pos_tagged


def lemmatize(text):
    tokens = preprocess(text)
    lemmatized = [token.lemma_ for token in tokens]
    return lemmatized


def remove_by_word(lemmatized, word_list):
    return [word for word in lemmatized if word not in word_list]


def get_pos_neg_words():
    def get_words(url):
        import requests
        words = requests.get(url).content.decode('latin-1')
        word_list = words.split('\n')
        index = 0
        while index < len(word_list):
            word = word_list[index]
            if ';' in word or not word:
                word_list.pop(index)
            else:
                index+=1
        return word_list
    #Get lists of positive and negative words
    p_url = 'http://ptrckprry.com/course/ssd/data/positive-words.txt'
    n_url = 'http://ptrckprry.com/course/ssd/data/negative-words.txt'
    positive_words = get_words(p_url)
    negative_words = get_words(n_url)
    return positive_words,negative_words


def do_pos_neg_sentiment_analysis(text_list,debug=False):
    positive_words,negative_words = get_pos_neg_words()
    results = list()
    for text in text_list:
        affirmative = negation_analysis(text)
        cpos = cneg = lpos = lneg = 0
        for word in generate_keywords(text):
            if word in positive_words:
                if debug:
                    print("Positive",word)
                cpos+=1
            if word in negative_words:
                if debug:
                    print("Negative",word)
                cneg+=1
        text_length = len(lemmatize(text))
        if text_length < 1:
            text_length = 1
        if affirmative:
            results.append( cpos/text_length - cneg/text_length )
        else:
            results.append( cneg/text_length - cpos/text_length )
    return results





def generate_keywords(text, word_list =[]):
    lemmatized = lemmatize(text)
    if len(word_list) > 0:
        lemmatized = remove_by_word(lemmatized, word_list)
    return lemmatized





def select_similar_emotions(user_txt, target_urls_txts, max_difference = 0.5, debug=False):
    result = []
    urls = []
    target_txts = []
    for url, txt in target_urls_txts.items():
        urls.append(url)
        target_txts.append(txt)
    target_scores = do_pos_neg_sentiment_analysis(target_txts)
    if debug:
        print(target_scores)
    user_score = do_pos_neg_sentiment_analysis([user_txt])[0]
    if debug:
        print("user score:",user_score)
    score_differences = [abs(target_score - user_score) for target_score in target_scores]
    result = [(urls[i], target_txts[i], score_differences[i]) for i in range ( len(target_txts) ) if score_differences[i] <= max_difference ]
    return result

    



def compare_keyword_similarity(user_txt, target_urls_txts, word_list = [], debug=False):
    valid_inds_texts = select_similar_emotions(user_txt, target_urls_txts)
    if len(valid_inds_texts) < 1:
        default_url = "url6" 
        default_target = (default_url, target_urls_txts[default_url])
        return (default_target)
    valid_texts = [text[1] for text in valid_inds_texts]
    texts = [generate_keywords(text, word_list) for text in valid_texts]
    dictionary = corpora.Dictionary(texts)
    feature_cnt = len(dictionary.token2id)
    corpus = [dictionary.doc2bow(text) for text in texts]
    tfidf = models.TfidfModel(corpus) 
    kw_vector = dictionary.doc2bow(generate_keywords(user_txt, word_list))
    index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features = feature_cnt)
    sim = index[tfidf[kw_vector]]
    if max(sim) == 0.00:
        #print("No similar videos")
        default_url = "url6" 
        default_target = (default_url, target_urls_txts[default_url])
        return (default_target)
    if debug:
        for i in range(len(sim)):
            print('keyword is similar to text%d: %.2f' % (i + 1, sim[i]))
    return(valid_inds_texts[np.argmax(sim)][:2])




         


if __name__ == '__main__':
    keyword = "I miss my parents so much."
    texts = {
        "url0":'Hey! Hope you will feel better soon',
         "url1":'Oh I know exam season is so tough. Hang in there!',
         "url2":'Congratulations for doing well in your midterm.',
         "url3":'I am so sorry about the midterm. I am sure you will do better next time.',
         "url4":'I know you are missing your family. You will see them soon.',
         "url5":"I know you are missing home. I'll bake your favorite cookies once you're back.",
         "url6":"Life can get super stressful, I get it. Remember that we all care about you. Stay Strong.",
         "url7":"We care about you."
         }
    sim_2 = compare_keyword_similarity(keyword, texts)
    print(sim_2)

    





