
from helper import remove_punc
import nltk
import nltk.tokenize as tk
import numpy as np

#Clean and lemmatize the contents of a document
#Takes in a file name to read in and clean
#Return a list of words, without stopwords and punctuation, and with all words stemmed
# NOTE: Do not append any directory names to doc -- assume we will give you
# a string representing a file name that will open correctly
def readAndCleanDoc(doc) :
    #1. Open document, read text into *single* string
    doc_text = open(doc)
    text = doc_text.read()
    doc_text.close()

    #2. Tokenize string using nltk.tokenize.word_tokenize

    nltk.download('punkt')
    words = tk.word_tokenize(text)
    #3. Filter out punctuation from list of words (use remove_punc)
    words = remove_punc(words)
    #4. Make the words lower case
    words = [word.lower() for word in words]
    #5. Filter out stopwords
    nltk.download('stopwords')
    stop = nltk.corpus.stopwords.words('english')
    words = [word for word in words if word not in stop]

    #6. Stem words
    stemmer = nltk.stem.PorterStemmer()
    words = [stemmer.stem(word) for word in words]

    return words
    
#Builds a doc-word matrix for a set of documents
#Takes in a *list of filenames*
#
#Returns 1) a doc-word matrix for the cleaned documents
#This should be a 2-dimensional numpy array, with one row per document and one 
#column per word (there should be as many columns as unique words that appear
#across *all* documents. Also, Before constructing the doc-word matrix, 
#you should sort the wordlist output and construct the doc-word matrix based on the sorted list
#
#Also returns 2) a list of words that should correspond to the columns in
#docword
def buildDocWordMatrix(doclist) :
    #1. Create word lists for each cleaned doc (use readAndCleanDoc)
    wordlist = [readAndCleanDoc(doc) for doc in doclist]
    docwords = wordlist;
    x = list(set(wordlist[0] + wordlist[1]))
    x.sort()
    #print(x) 
    xlen = len(x)

    docwords = np.zeros((len(doclist), len(x)))
    #print(xlen)

    wordmap = {}; 
    for word in x:
        wordmap[word] = [0, 0]

    for i in range(len(wordlist)):
        for word in wordlist[i]:
            wordmap[word][i] += 1
    
    for i, key in enumerate(wordmap.keys()):
        for j, value in enumerate(wordmap[key]):
            docwords[j, i] = value
    #2. Use these word lists to build the doc word matrix
    #print(docwords)
    wordlist = x[:]
    return docwords, wordlist
    
#Builds a term-frequency matrix
#Takes in a doc word matrix (as built in buildDocWordMatrix)
#Returns a term-frequency matrix, which should be a 2-dimensional numpy array
#with the same shape as docword
def buildTFMatrix(docword) :
    #fill in
    sums = np.sum(docword, axis = 1)
    tf = docword[:,:]
    for i, value in enumerate(sums):
        tf[i, :] = tf[i, :]/value
    return tf
    
#Builds an inverse document frequency matrix
#Takes in a doc word matrix (as built in buildDocWordMatrix)
#Returns an inverse document frequency matrix (should be a 1xW numpy array where
#W is the number of words in the doc word matrix)
#Don't forget the log factor!
def buildIDFMatrix(docword) :
    #fill in
    idf = np.sum(np.ceil(docword/np.max(docword)), axis = 0)
    idf = np.log10((np.ones_like(idf)*(docword.shape[0]))/idf)
    return np.expand_dims(idf, axis=0)
    
#Builds a tf-idf matrix given a doc word matrix
def buildTFIDFMatrix(docword) :
    #fill in
    tf = buildTFMatrix(docword)
    idf = buildIDFMatrix(docword)
    tfidf = tf[:,:]
    for i in range(int(tf.shape[0])):
        tfidf[i,:] = tfidf[i,:]*idf 
    return tfidf
    
#Find the three most distinctive words, according to TFIDF, in each document
#Input: a docword matrix, a wordlist (corresponding to columns) and a doclist 
# (corresponding to rows)
#Output: a dictionary, mapping each document name from doclist to an (ordered
# list of the three most common words in each document
def findDistinctiveWords(docword, wordlist, doclist) :
    distinctiveWords = {}
    #fill in
    #you might find numpy.argsort helpful for solving this problem:
    #https://docs.scipy.org/doc/numpy/reference/generated/numpy.argsort.html
    tfidf = buildTFIDFMatrix(docword)
    unique = 3
    for path in doclist:
        distinctiveWords[path] = []
        #wordlist.append([])

    for i in range(unique):
        maxim = np.argmax(tfidf, axis = 1)
        for j,key in enumerate(distinctiveWords.keys()):
            distinctiveWords[key].append(wordlist[maxim[j]])
            tfidf[j, maxim[j]] = 0
    print(maxim)
    
    #pass
    return distinctiveWords


if __name__ == '__main__':
    from os import listdir
    from os.path import isfile, join, splitext
    
    ### Test Cases ###
    directory='lecs'
    path1 = join(directory, '1_vidText.txt')
    path2 = join(directory, '2_vidText.txt')
    
    # Uncomment and recomment ths part where you see fit for testing purposes
    
    #print("*** Testing readAndCleanDoc ***")
    #print(readAndCleanDoc(path1)[:])
    print("*** Testing buildDocWordMatrix ***") 
    doclist =[path1, path2]
    docword, wordlist = buildDocWordMatrix(doclist)
    print(docword.shape)
    print(len(wordlist))
    print(docword[0][0:10])
    print(wordlist[0:10])
    print(docword[1][0:10])
    # print("*** Testing buildTFMatrix ***") 
    # tf = buildTFMatrix(docword)
    # print(tf[0][0:10])
    # print(tf[1][0:10])
    # print(tf.sum(axis =1))
    # print("*** Testing buildIDFMatrix ***") 
    # idf = buildIDFMatrix(docword)
    # print(idf[0][0:10])
    # print("*** Testing buildTFIDFMatrix ***") 
    # tfidf = buildTFIDFMatrix(docword)
    # print(tfidf.shape)
    # print(tfidf[0][0:10])
    # print(tfidf[1][0:10])
    # print(tfidf)
    print("*** Testing findDistinctiveWords ***")
    print(findDistinctiveWords(docword, wordlist, doclist))
    
