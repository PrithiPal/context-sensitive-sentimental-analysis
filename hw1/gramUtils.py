
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
	treelist=[]
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
				if len(onesentence) > 0:
					treelist[-1] = treelist[-1] + onesentence
					treelist.append(currentline)
					# treeset = treeset + onesentence + '\n' + currentline 
					onesentence = ""
				else:
					treelist.append(currentline) # = treeset + onesentence + '\n' + currentline 

	# eof (last tree) -- tedious algorithm but whatever (for now)
	if len(onesentence) > 0:
		treelist[-1] = treelist[-1] + onesentence
		onesentence = ""

	return treelist

# usage:  makeRulesFromTreeList(getRulesFromDevTrees("devset.tree"))
def makeRulesFromTreeList(lstGroupOfTrees):
	lstRules=[]
	for item in lstGroupOfTrees:
		tr = Tree.fromstring(item)
		lstRules.append(tr.productions())

	return lstRules