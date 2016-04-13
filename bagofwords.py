#Author: Li Zhao
#Pomona College 2016 Computer Science Senior Project

#This includes ALL of the code that is currently in use.
#Version: 03/17/2016
#Search for "TODO"

# Comments:
# Functions like this: thisIsFunction
# Variables like this: thisVariable_type 
# 'Simple' Variables: num###, path
# s = string, l = list, ll = 2d list, df = dataframe 

# Contents:
# IMPORTS
# OBJECTS
# FUNCTIONS
# 1. scriptToWords
# 2. scriptLength
# 3. findPercentOfInterior
# 4. toLines
# 5. characters
# 6. gender (generates female and neutral)
# 7. offensive
# 8. tech
# 9. dialog
# 10. description
# MAIN
# 1. Prepocessing text
# 2. Bag of words
# 3. Script length 
# 4. Random forest, bayes, svm
# 5. Testing and output
# ARCHIVED



# References: 
# https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-1-for-beginners-bag-of-words
# https://www.cs.cmu.edu/Groups/AI/areas/nlp/corpora/names/0.html
# http://www.cs.cmu.edu/~biglou/resources/bad-words.txt
# https://en.wikipedia.org/w/index.php?title=Cinematic_techniques&printable=yes



##########################################
"""_____________IMPORTS______________"""
##########################################


from __future__ import division
import csv
import itertools
import pandas as pd 
import numpy as np
import re
import matplotlib.pyplot as plt
# Import the stop word list         
import nltk
#nltk.download()
from nltk.corpus import stopwords 
stops = set(stopwords.words("english")) 
import os, os.path

from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus.reader.plaintext import PlaintextCorpusReader
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from collections import Counter

import pickle

##########################################
"""_____________Objects______________"""
##########################################

class Line:

      def __init__(self,str_s):

        self.str = str_s                
        # 99999 is uninitialized
        # id is index number of line
        self.id = 99999
        # 0 is character, 1 is dialogue, 3 is scener header, 4 is description 
        self.type =99999 
        # which scene line belongs to 
        self.scence = 99999

        self.speaker = ""
        self.listeners = []

      #debugging
      def printAll(self):
        print(self.str)
        print("id:"+str(self.id))
        print("type:"+str(self.type))
        print("scene:"+str(self.scence))
        print("speaker:"+ self.speaker)
        print("listeners:")
        print(self.listeners)
        


      # def setId(self, idNo_n):
      #   self.scene = idNo_n

      # def setType(self, typeNo_n):
      #   self.scene = typeNo_n

      # def setDialogue(self,speaker_s, listeners_l):
      #   self.speaker = speaker_s
      #   self.listeners = listeners_l 

      # def setScene(self, sceneNo_n):
      #   self.scene = sceneNo_n






##########################################
"""_____________FUNCTIONS______________"""
##########################################



def scriptToWords(scriptText_s):

	# Regex to remove all that's not letters
	lettersOnly_s = re.sub("[^a-zA-Z]",           # The pattern to search for
	                       " ",                   # The pattern to replace it with
	                       scriptText_s)  # The text to search

	# All to lowercase and split into words
	lowercase_s = lettersOnly_s.lower()        
	words_l = lowercase_s.split()              

	#    Remove stop words TODO: do ucorpus use more stop words
	#    u means unicode
	meaningfulWords_l = [w for w in words_l if not w in stops]


	#   Join the words back into one string separated by space 
	return( " ".join( meaningfulWords_l ))   





def scriptLength(scriptText_s):
    return len(scriptText_s.split())   




def findPercentOfInterior(scriptText_s):
    numInt_n = scriptText_s.count('INT')
    numExt_n = scriptText_s.count('EXT')
    if ((numInt_n+numExt_n) == 0): 
        return 0
    else: 
        return round((numInt_n/(numInt_n+numExt_n)), 2)


def numScences(scriptText_s):
    return (scriptText_s.count('INT') + scriptText_s.count('EXT'))




# Helper function 
def toLines(scriptName_s):
    lines_l = []
    with open(scriptName_s, "r") as f:
        for line in f:
            x = Line(line)
            lines_l.append(x)
    #print lines_l 
    return lines_l 


# 1. Picks up on the screenplay title
# 2. Doesn't do punctuation / numbers
# 3. Doesn't do \n\t NAME \n
# Also marks headers for scences
def characters(lines_l, scriptName_s ):
    allChar_l = []
    index_n = 0
    scence_n = 0
    for line_s in lines_l:
        x = re.findall('^\s*[A-Z]+(?=\s*$)', line_s.str)
        #x = re.findall('^ *\t*\r*\f*\v*[A-Z]+(?= *\t*\r*\f*\v*\n)', line_s.str)
        # if character
        if x!=[]:
            allChar_l.append(x[0].strip())
            line_s.type =0 # character
        else:
            y = re.findall('INT.*|EXT.*', line_s.str)
            if y!=[]:
                line_s.type = 2 # header
                scence_n = scence_n+1

        line_s.id = index_n
        line_s.scence = scence_n
        index_n = index_n+1
        # line_s.printAll()


    #print set(allChar_l)

    moreThanOne_l = [k for k, v in Counter(allChar_l).iteritems() if v > 1 ]
    if len(moreThanOne_l) ==0:
        raise NameError('No characters found:'+ scriptName_s)
    return set(moreThanOne_l)


# 1. between char and char(check for parentheses)
# 2. between char and punctuation, if there are no more letters before end of line, 
#    else continue to search for punctuation on next line.
# 3. This misses the case when a line ends with an ending punctuation, and there is more dialog on the 
#    next line.
def dialogue(lines_l):
    numDialog =0
    for i in range(0, len(lines_l)):
        if ((lines_l[i].type == 0 )and ((i+1) < len(lines_l))):
            j = i +1 #|(.*\.\.\.)|(.*\-)
            while ((j < len(lines_l)) and (re.findall('^((.*\.)|(.*\?)|(.*!))(\s)*$',lines_l[j].str) == [])):
                lines_l[j].type = 1 #dialogue
                j= j+1
                numDialog = numDialog +1 #lines_l[j].printAll()
            if ((j < len(lines_l))):
                lines_l[j].type = 1 #dialogue
            numDialog = numDialog +1
            i = j+1
        else:
            i = i+1 
    percentageOfDialog_n = numDialog/len(lines_l)
    return percentageOfDialog_n

def description(lines_l):
    numDes = 0
    for i in range(0, len(lines_l)):
        if ((lines_l[i].type == 99999) and \
            (re.findall('^(\s)*$',lines_l[i].str)==[])):
            lines_l[i].type = 3 #description
            numDes = numDes+1
        i = i +1
    percentageOfDes_n = numDes/len(lines_l)
    return percentageOfDes_n



# 1. Weak, may be biased towards English names, since that is the filter used. 
#    Can be improved with good use of the US census data
def gender(characters_l):
    female_l =  open("./female.txt", "r").read().lower().split()
    male_l =  open("./male.txt", "r").read().lower().split()
    numCharacters_n = len(characters_l)
    female_n = 0
    male_n = 0
    neutral_n = 0
    for name_s in characters_l:
        character_s = name_s.lower()
        if character_s in female_l and character_s in male_l:
            neutral_n = neutral_n +1
        elif character_s in female_l:
            #print "FOUND FEMALE:" + character_s
            female_n = female_n +1
            #print female_n
        elif character_s in male_l:
            male_n = male_n +1
        else:
            neutral_n = neutral_n +1

    return (female_n/numCharacters_n, neutral_n/numCharacters_n)

# 1. This is made with stopwords left in, different from scriptToWords.
def offensive(scriptText_s):
    offensive_l =  open("./offensive.txt", "r").read().lower().split()
    allWords_l = scriptText_s.lower().split()
    offensive_n = 0
    for word_s in allWords_l:
        if word_s in offensive_l:
            offensive_n = offensive_n +1
            #print word 

    return (offensive_n/len(allWords_l)) 

# 1. Weak, can be improved by adding to filmTechWords_l
def tech(scriptText_s):
    allWords_l = scriptText_s.split()
    tech_n = 0
    for word_s in filmTechWords_l:
        if word_s in scriptText_s:
            tech_n = tech_n +1
            #print word 

    return (tech_n/len(allWords_l)) 
    



##########################################
"""_____________MAIN______________"""
##########################################
#corpusdir = 'mini/' # Directory of corpus.

#newcorpus = PlaintextCorpusReader(corpusdir, '.*') #making corpus

#make a csv file
#csvfile = open('dump.csv', 'w')
#wr = csv.writer(csvfile, quoting=csv.QUOTE_NONE)#  delimiter=','
#wr.writerow(["Title", "Bagofwords", "Produced"])

# Put directory's scripts in a list
#path = "./mini/"
path = "./training/"
scripts_l = [(path + name) for name in os.listdir(path) if os.path.isfile(path + name) and name != ".DS_Store"]
#scriptsNameMatching_l = [(path + name) for name in os.listdir(path) if os.path.isfile(path + name) and name != ".DS_Store"]
#print scripts

# Get the number of scripts based on the dataframe column size
numScripts = len(scripts_l)

# Initialize an empty list to hold the clean scripts
cleanScripts_l = []

# Initialize an empty list to hold script length
sizeOfScripts_l = []

# Initialize an empty list for findPercentOfInterior
percentOfInterior_l = []


# Initialize an empty list for numScences
numScences_l = []

# Initialize an empty list for numChar
numChar_l = []

# Initialize an empty list for percentage female
female_l = []
# Initialize an empty list for percentage neutral
neutral_l = []

# Initialize an empty list for possibly offensive words
offensive_l = []

# Initialize an empty list for tech words
tech_l = []

# Make list of filmTechWords
filmTechWords_l =['SINGLE-CAMERA SETUP', 'SEQUENCE SHOT', 'TRUNK SHOT', \
                'REVERSE ANGLE', 'FILMMAKING', 'LOW-KEY LIGHTING', \
                'MEDIUM CLOSE-UP', 'PAN', 'DOLLYING', 'CAMERA ANGLE', \
                'SMPTE TIMECODE', 'FX', '3D COMPUTER GRAPHICS', 'FADE IN/OUT', \
                'ELLIPSIS', 'TILT SHOT', 'FILL LIGHT', 'GAZE/LOOK', \
                'MOOD LIGHTING', 'DEEP FOCUS', 'CAMERA TRACKING', 'STORY BOARD', \
                'SPLIT EDIT', 'SHAKY CAM', 'CUTAWAY', 'TALKING HEAD', \
                'FLOOD LIGHTING', 'DOLLY', 'MATCH CUT', 'HEAD SHOT', \
                'MEDIUM SHOT', 'AERIAL SHOT', '3D FILM FOR MOVIE HISTORY', \
                'LS', 'LOW-ANGLE SHOT', 'ANGLE OF VIEW', 'PULL BACK SHOT', \
                'CUT TO', 'CONTINUITY CUTS', 'HANGING MINIATURE', 'KEYING',\
                'TILT', 'VIDEO PRODUCTION', 'STEADICAM', 'EXTREME LONG SHOT',\
                'HIGH-ANGLE SHOT', 'FULL SHOT', 'DISSOLVE', 'BOOM SHOT', \
                'CLOSE-UP', 'MONTAGE', 'SCREEN DIRECTION', 'HIGH-KEY LIGHTING',\
                'FILM FRAME', 'STALKER VISION', "WORM'S-EYE VIEW", 'WHIP PAN',\
                'MULTIPLE-CAMERA SETUP', 'OPTICAL EFFECTS', 'EXTREME CLOSE-UP',\
                'FULL FRAME', 'ESTABLISHING SHOT', 'TRAVELLING SHOT', 'A-ROLL',\
                'STAGE LIGHTING', 'ZOOM', 'FORCED PERSPECTIVE', 'CRANE SHOT',\
                'BRIDGING SHOT', 'WIPE', "BIRD'S EYE SHOT", 'DUTCH ANGLE', \
                'KEY LIGHT', 'COMPUTER-GENERATED IMAGERY', 'CROSS-CUTTING',  \
                'SCENE', 'MONEY SHOT', 'TOP-DOWN PERSPECTIVE', 'CHROMA KEY',  \
                'ECU', 'CU', 'FOLLOW SHOT', 'MCU', 'DISSOLVE TO', 'B-ROLL', \
                'CUT', 'FADE IN', 'CAMEO LIGHTING', 'TRACKING SHOT', \
                'SHOT REVERSE SHOT', 'AERIAL PERSPECTIVE', 'REMBRANDT LIGHTING',\
                'OVER THE SHOULDER SHOT', 'SOFT LIGHT', 'EYE-LINE MATCHING', \
                'IRIS IN/OUT', 'KEY LIGHTING', 'AMERICAN SHOT', 'JUMP CUT', \
                'FOCUS IN', 'L CUT', 'LONG SHOT', 'DIGITAL COMPOSITING', 'ELS',\
                'MASTER SHOT', 'CAMERA OPERATOR', 'V.O.', 'FOCUS', 'CAMERA DOLLY',\
                'LENS FLARE', 'DOLLYING SHOT', 'POINT OF VIEW SHOT', 'PANNING', \
                "BIRD'S-EYE VIEW", 'TWO SHOT', 'SMASH CUT', 'INSERT', 'FADE OUT', \
                'SUBJECTIVE CAMERA', 'FRAMING', 'REACTION SHOT', 'RACK FOCUSING', \
                'SHOT', 'MS', 'FOCUS OUT', 'SLOW CUTTING', 'FAST CUTTING', \
                'SPLIT SCREEN', 'MLS', 'LONG TAKE', 'DOLLY ZOOM', 'SPECIAL EFFECTS', \
                'VOICE OVER', 'BACKGROUND LIGHTING', '(V.O.)', 'FREEZE-FRAME SHOT', \
                'MEDIUM LONG SHOT', 'TITLE OVER', 'ONE SHOT', 'BLUESCREEN', \
                'CAMERA COVERAGE', 'VOICE-OVER', 'FLASHFORWARD', 'WALK AND TALK', \
                'TAKE', 'LAP-DISSOLVE', 'BULLET TIME', 'FLASHBACK', 'EDITING', 'DIEGESIS']

percentageOfDialog_l = []
percentageOfDes_l =[]

# Loop over each review; create an index i that goes from 0 to the length
# of the script list 
for i in xrange( 1, numScripts+1):
    # Call our function for each one, and add the result to the list of
    # clean scripts
    scriptName_s = scripts_l[i-1]
    scriptText_s = open(scriptName_s, "r").read()
    scriptLines_l = toLines(scriptName_s)  

    cleanScripts_l.append( scriptToWords(scriptText_s ) )
    sizeOfScripts_l.append(scriptLength(scriptText_s))
    percentOfInterior_l.append(findPercentOfInterior(scriptText_s))
    numScences_l.append(numScences(scriptText_s)) 

    characters_l = characters(scriptLines_l, scriptName_s )
    numChar_l.append(len(characters_l)) 
    #female_l.append(gender(characters_l)[0])
    neutral_l.append(gender(characters_l)[1])
    offensive_l.append(offensive(scriptText_s))
    tech_l.append(tech(scriptText_s))
    percentageOfDialog_l.append(dialogue(scriptLines_l))
    percentageOfDes_l.append(description(scriptLines_l))


    #print scriptToWords( scriptToWords( "./alpha/" + scripts[i]))
    #wr.writerow([scripts[i],1])#scriptToWords( scripts[i]), 1])
    print "Cleaned " + str(i) + " scripts."


##################################################
# Bag of words                                   #
##################################################



# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool.  
vectorizer_cv = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000) 

# fit_transform() does two functions: First, it fits the model
# and learns the vocabulary; second, it transforms our training data
# into feature vectors. The input to fit_transform should be a list of 
# strings.
trainDataFeatures_v = vectorizer_cv.fit_transform(cleanScripts_l)
#print vectorizer.get_feature_names()
# Numpy arrays are easy to work with, so convert the result to an 
# array
trainDataFeatures_a = trainDataFeatures_v.toarray()
features_df= pd.DataFrame(trainDataFeatures_a)
#testing
#vocab = vectorizer.get_feature_names()


##################################################
# length of scripts                              #
##################################################
length_df = pd.DataFrame(sizeOfScripts_l)
#print length_df
features_df = features_df.assign(scriptsLen =length_df)
#train_data_features = train_data_features.append(sizeof_scripts)


##################################################
# percentOfInterior                              #
##################################################
#print "percentOfInterior:"
#print percentOfInterior_l
percentOfInterior_df = pd.DataFrame(percentOfInterior_l)
features_df = features_df.assign(percentOfInterior =percentOfInterior_df)




##################################################
# numScences                                     #
##################################################


numScences_df = pd.DataFrame(numScences_l)
features_df = features_df.assign(numScences =numScences_df)


##################################################
# numChar                                        #
##################################################


numChar_df = pd.DataFrame(numChar_l)
features_df = features_df.assign(numChar =numChar_df)



# ##################################################
# # gender                                         #
# ##################################################


# female_df = pd.DataFrame(female_l)
# features_df = features_df.assign(female =female_df)
neutral_df = pd.DataFrame(neutral_l)
features_df = features_df.assign(neutral =neutral_df)


##################################################
# offensive                                      #
##################################################

offensive_df = pd.DataFrame(offensive_l)
features_df = features_df.assign(offensive =offensive_df)

##################################################
# tech                                           #
##################################################

tech_df = pd.DataFrame(tech_l)
features_df = features_df.assign(tech =tech_df)
percentageOfDialog_df = pd.DataFrame(percentageOfDialog_l)
features_df = features_df.assign(percentageOfDialog =percentageOfDialog_df)

percentageOfDes_df = pd.DataFrame(percentageOfDes_l)
features_df = features_df.assign(percentageOfDes =percentageOfDes_df)




print features_df

##################################################
# pickling                                       #
##################################################


features_file = open('features.pkl', 'wb')

# Pickle dictionary using protocol 0.
pickle.dump(features_df, features_file)


features_file.close()



##################################################
# Random Forest                                  #
##################################################

#Unpickling

features_file = open('features.pkl', 'rb')

features_df = pickle.load(features_file)

features_file.close()


#make a label document
"""
csvfile = open('labels.csv', 'w')
wr = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
wr.writerow(["Title", "Produced"])
for i in xrange( 1, numScripts):
	wr.writerow([scripts[i],1])"""     


produced_l = [name for name in os.listdir("../../backup-partitioned-corpus/pcorpus/training") if name != ".DS_Store"]
unproduced_l = [name for name in os.listdir("../../backup-partitioned-corpus/ucorpus/training") if name != ".DS_Store"]
names_l = [name for name in os.listdir(path) if name != ".DS_Store"]
train_ll = [[0 for x in range(2)] for x in range(numScripts)] 

#train[0][0] = "Name"
#train[0][1] = "Produced"

for i in range(numScripts): 
    thisName_s = names_l[i]
    train_ll[i][0] = thisName_s
    train_ll[i][1] = 0 if thisName_s in unproduced_l else 1

#print train_ll

#print train

#write a labels csv file
"""
csvfile = open('labels.csv', 'w')
wr = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
wr.writerow(["Title", "Produced"])
for i in xrange( numScripts):
    wr.writerow(train[i])
"""
headers_l = ["Title", "Produced"]
label_df = pd.DataFrame(train_ll, columns=headers_l)


#csvfile = open('labels.csv', 'w')
#df.to_csv("labelstest.csv", quoting=2,header=0, delimiter="," )


#finaltrain = pd.read_csv("labelstest.csv", header=0, delimiter=",", quoting=2) #encoding='utf-16',
#finaltrain = open('labels.csv', 'r').read()
print "label_df:" 
print label_df
print "Training the random forest..."


# Initialize a Random Forest classifier with 100 trees
# Fit the forest to the training set, using the bag of words as 
# features and the sentiment labels as the response variable
forest_f = RandomForestClassifier(n_estimators = 100) 
forest_f = forest_f.fit( features_df, label_df["Produced"] )

#fit a bayes classfier
bayes_gnb = MultinomialNB()
bayes_gnb = bayes_gnb.fit(features_df, label_df["Produced"] )

#fit a svm classfier
svm_svc = LinearSVC()
svm_svc = svm_svc.fit(features_df, label_df["Produced"] ) 

#"""
##################################################
# Testing                                        #
##################################################


# Read the test data
testpath_s = "./testing/"
testcorpus_c = PlaintextCorpusReader(testpath_s, '.*') #making corpus

tests_l = [(testpath_s+ tname_s) for tname_s in os.listdir(testpath_s) if  tname_s != ".DS_Store"]
testsNameMatching_l = [tname_s for tname_s in os.listdir(testpath_s) if tname_s != ".DS_Store"]

#print tests_l

# Create an empty list and append the clean reviews one by one
numTests_n = len(tests_l)
cleanScriptsTests_l = []

# Initialize an empty list to hold script length
sizeOfScriptsTests_l = []

# Initialize an empty list for findPercentOfInterior
percentOfInteriorTests_l = []

# Initialize an empty list for numScences
numScencesTests_l = []

# Initialize an empty list for numChar
numCharTests_l = []

# Initialize an empty list for percentage female
femaleTests_l = []
# Initialize an empty list for percentage neutral
neutralTests_l = []

# Initialize an empty list for possibly offensive words
offensiveTests_l = []

# Initialize an empty list for tech words
techTests_l = []

percentageOfDialogTests_l = []
percentageOfDesTests_l = []


print "Cleaning and parsing the test set ...\n"
for i in xrange(0,numTests_n):
#    if( (i+1) % 1000 == 0 ):
        print "Test %d of %d\n" % (i+1, numTests_n)


        scriptNameTests_s = tests_l[i-1]
        scriptTextTests_s = open(scriptNameTests_s, "r").read()
        scriptLinesTests_l = toLines(scriptNameTests_s)  

        cleanScriptsTests_l.append( scriptToWords(scriptTextTests_s ) )
        sizeOfScriptsTests_l.append(scriptLength(scriptTextTests_s))
        percentOfInteriorTests_l.append(findPercentOfInterior(scriptTextTests_s))
        numScencesTests_l.append(numScences(scriptTextTests_s)) 

        charactersTests_l = characters(scriptLinesTests_l, scriptNameTests_s )
        numCharTests_l.append(len(charactersTests_l)) 
        #femaleTests_l.append(gender(charactersTests_l)[0])
        neutralTests_l.append(gender(charactersTests_l)[1])
        offensiveTests_l.append(offensive(scriptTextTests_s))
        techTests_l.append(tech(scriptTextTests_s))
        percentageOfDialogTests_l.append(dialogue(scriptLinesTests_l))
        percentageOfDesTests_l.append(description(scriptLinesTests_l))




# Get a bag of words for the test set, and convert to a numpy array
testFeatures_v = vectorizer_cv.transform(cleanScriptsTests_l)
testFeatures_a = testFeatures_v.toarray()



featuresTests_df= pd.DataFrame(testFeatures_a)
#testing
#vocab = vectorizer.get_feature_names()


##################################################
# length of scripts                              #
##################################################
lengthTests_df = pd.DataFrame(sizeOfScriptsTests_l)
#print length_df
featuresTests_df = featuresTests_df.assign(scriptsLen =lengthTests_df)
#train_data_features = train_data_features.append(sizeof_scripts)


##################################################
# percentOfInterior                              #
##################################################
#print "percentOfInterior:"
#print percentOfInterior_l
percentOfInteriorTests_df = pd.DataFrame(percentOfInteriorTests_l)
featuresTests_df = featuresTests_df.assign(percentOfInterior =percentOfInteriorTests_df)




##################################################
# numScences                                     #
##################################################


numScencesTests_df = pd.DataFrame(numScencesTests_l)
featuresTests_df = featuresTests_df.assign(numScences =numScencesTests_df)


##################################################
# numChar                                        #
##################################################


numCharTests_df = pd.DataFrame(numCharTests_l)
featuresTests_df = featuresTests_df.assign(numChar =numCharTests_df)



##################################################
# gender                                         #
##################################################


# femaleTests_df = pd.DataFrame(femaleTests_l)
# featuresTests_df = featuresTests_df.assign(female =femaleTests_df)
#print("FEMALE:")
#print femaleTests_df
neutralTests_df = pd.DataFrame(neutralTests_l)
featuresTests_df = featuresTests_df.assign(neutral =neutralTests_df)


##################################################
# offensive                                      #
##################################################

offensiveTests_df = pd.DataFrame(offensiveTests_l)
featuresTests_df = featuresTests_df.assign(offensive =offensiveTests_df)

##################################################
# tech                                           #
##################################################

techTests_df = pd.DataFrame(techTests_l)
featuresTests_df = featuresTests_df.assign(tech =techTests_df)



percentageOfDialogTests_df = pd.DataFrame(percentageOfDialogTests_l)
featuresTests_df = featuresTests_df.assign(percentageOfDialog =percentageOfDialogTests_df)

percentageOfDesTests_df = pd.DataFrame(percentageOfDesTests_l)
featuresTests_df = featuresTests_df.assign(percentageOfDes =percentageOfDesTests_df)








# Use the random forest to make predictions
result_l = forest_f.predict(featuresTests_df)
actual_l = []

producedTests_l = [name for name in os.listdir("../../backup-partitioned-corpus/pcorpus/testing") if name != ".DS_Store"]
unproducedTests_l = [name for name in os.listdir("../../backup-partitioned-corpus/ucorpus/testing") if name != ".DS_Store"]




for testName_s in testsNameMatching_l: 
    if (testName_s in unproducedTests_l):
        actual_l.append(0)
    elif (testName_s in producedTests_l):
        actual_l.append(1)
    else:
        actual_l.append(999) 

# Copy the results to a pandas dataframe with an "title" column and
# a "produced" column
accuracy_l = []
truecount_n =0
for i in range(0, len(result_l)):
    if result_l[i] == actual_l[i]:
        accuracy_l.append("TRUE")
        truecount_n = truecount_n+1
    else:
        accuracy_l.append("FALSE")

output = pd.DataFrame( data={"title":testsNameMatching_l,  "actual_produced":actual_l, "RF_predicted_produced":result_l,"RF_accuracy": accuracy_l} )
print ("Random Forest Accuracy:" + str(truecount_n/len(result_l)) )


#Bayes prediction

resultBayes_l = bayes_gnb.predict(featuresTests_df)
accuracyBayes_l = []
truecountBayes_n =0
for i in range(0, len(resultBayes_l)):
    if resultBayes_l[i] == actual_l[i]:
        accuracyBayes_l.append("TRUE")
        truecountBayes_n = truecountBayes_n+1
    else:
        accuracyBayes_l.append("FALSE")
resultBayes_df = pd.DataFrame(resultBayes_l)
accuracyBayes_df = pd.DataFrame(accuracyBayes_l)
output = output.append(resultBayes_df)
output = output.append(accuracyBayes_df)
print ("Bayes Accuracy:" + str(truecountBayes_n/len(resultBayes_l)) )


#SVM prediction
resultSVM_l = svm_svc.predict(featuresTests_df)
accuracySVM_l = []
truecountSVM_n =0
for i in range(0, len(resultSVM_l)):
    if resultSVM_l[i] == actual_l[i]:
        accuracySVM_l.append("TRUE")
        truecountSVM_n = truecountSVM_n+1
    else:
        accuracySVM_l.append("FALSE")
resultSVM_df = pd.DataFrame(resultSVM_l)
accuracySVM_df = pd.DataFrame(accuracySVM_l)
output = output.append(resultSVM_df)
output = output.append(accuracySVM_df)
print ("SVM Accuracy:" + str(truecountSVM_n/len(resultSVM_l)) )


#feature selection


featureNames_l = vectorizer_cv.get_feature_names()
manualFeatureNames_l = list(featuresTests_df.columns.values)



print "Forest features:"
goodFeaturesForest_l = []
indsForest_l= forest_f.feature_importances_.tolist()
#print indsForest_l

# for i in range(0, len(indsForest_l)):
#     if (indsForest_l[i]> np.mean(indsForest_l)) and (i >= 5000):
#         goodFeaturesForest_l.append(manualFeatureNames_l[i])

#     elif (indsForest_l[i]> np.mean(indsForest_l)):
#         #print i
#         goodFeaturesForest_l.append(featureNames_l[i])

# print goodFeaturesForest_l    

for i in range(0, len(indsForest_l)):
    if (i >= 5000):
        goodFeaturesForest_l.append((indsForest_l[i], manualFeatureNames_l[i]))

    else: goodFeaturesForest_l.append((indsForest_l[i],featureNames_l[i]))
goodFeaturesForest_l = sorted(goodFeaturesForest_l, reverse=True)
print goodFeaturesForest_l    

goodFeaturesBayes_l = []
indsBayes_l = np.argsort(bayes_gnb.coef_[0, :])[-50:]
print "Bayes features:" 
#print indsBayes_l
for i in indsBayes_l:
    if i >= 5000:
        goodFeaturesBayes_l.append(manualFeatureNames_l[i])
    else:
        goodFeaturesBayes_l.append(featureNames_l[i])
print goodFeaturesBayes_l 

goodFeaturesSVM_l = []
indsSVM_l = np.argsort(svm_svc.coef_[0, :])[-50:]
print "SVM features:" 
#print indsSVM_l
for i in indsSVM_l:
    if i >= 5000:
        goodFeaturesSVM_l.append(manualFeatureNames_l[i])
    else:
        goodFeaturesSVM_l.append(featureNames_l[i])
print goodFeaturesSVM_l 
#######################################################
# Use pandas to write the comma-separated output file #
#######################################################
output.to_csv( "output.csv", index=False, quoting=3)

#"""










##########################################
"""_____________ARCHIVED______________"""
##########################################






