
import nltk
from nltk import Tree

def stripNewLines(strTree):
	# count=0
	partStr=""
	isNewLineFound = True
	while isNewLineFound:
		partStr = strTree.partition('\n')
		print(partStr)
		strTree = partStr[0] + partStr[2]
		isNewLineFound = len(partStr[2]) != 0
		print("New str: ", strTree, '\n')
		# if strTree[count] == '\n':
		# print (strTree[count])
		# count=count+1
	print(strTree)


def getRulesFromDevTrees(strFilename):
	treeset=""
	onesentence=""
	with open(strFilename) as file:
		lines = file.readlines()
		# print(lines, "TYPE", type(lines))
		for currentline in lines:
			# print ("=WITHout PART:= " , currentline)
			currentline = currentline.partition("\n")[0]
			# print ("=WITH PART:= " , currentline)
			if currentline[0].isspace():
				onesentence = onesentence + " " + currentline.strip()
			else:
				treeset = treeset + onesentence + '\n' + currentline 
				onesentence = ""

	return treeset