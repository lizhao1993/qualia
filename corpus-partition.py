# This checks the training directory with the backup directory to ensure we have the correct files and the correct number of files.


import os, os.path

# path = "./training/"
# produced_l = [name for name in os.listdir("../../backup-partitioned-corpus/pcorpus/training") if name != ".DS_Store"]
# unproduced_l = [name for name in os.listdir("../../backup-partitioned-corpus/ucorpus/training") if name != ".DS_Store"]
path = "./testing/"
produced_l = [name for name in os.listdir("../../backup-partitioned-corpus/pcorpus/testing") if name != ".DS_Store"]
unproduced_l = [name for name in os.listdir("../../backup-partitioned-corpus/ucorpus/testing") if name != ".DS_Store"]


names_l = [name for name in os.listdir(path) if name != ".DS_Store"]

scripts_l = [(path + name) for name in os.listdir(path) if os.path.isfile(path + name) and name != ".DS_Store"]
#scriptsNameMatching_l = [(path + name) for name in os.listdir(path) if os.path.isfile(path + name) and name != ".DS_Store"]
#print scripts

# Get the number of scripts based on the dataframe column size
numScripts = len(scripts_l)
train_ll = [[0 for x in range(2)] for x in range(numScripts)] 

#train[0][0] = "Name"
#train[0][1] = "Produced"
actualproduced = 0
actualunproduced =0
for i in range(numScripts): 
    thisName_s = names_l[i]
    train_ll[i][0] = thisName_s
    if thisName_s in unproduced_l:
    	train_ll[i][1] = 0 
    	actualunproduced = actualunproduced +1 
    elif thisName_s in produced_l:
    	train_ll[i][1] = 0 
    	actualproduced = actualproduced +1 
    else: print ("Not in backup:") + thisName_s

for i in range(len(unproduced_l)): 
	thisName_s = unproduced_l[i]
	if (thisName_s not in names_l):
		print ("Unproduced not in testing:") + thisName_s

for i in range(len(produced_l)): 
	thisName_s = produced_l[i]
	if (thisName_s not in names_l):
		print ("Produced not in testing:") + thisName_s

#print train_ll
#print "Checking training corpus...."
print "produced should have" 
print (len(produced_l))
print "unproduced should have" 
print (len(unproduced_l))
print "produced has" 
print (actualproduced)
print "unproduced has" 
print (actualunproduced)
