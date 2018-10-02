
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


def getTreesFromDevset(strFilename):
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

# usage:  makeRulesFromTreeList(getTreesFromDevset("devset.trees"))
def makeRulesFromTreeList(lstGroupOfTrees, isDuplicate=False):
	lstRules=[]
	for item in lstGroupOfTrees:
		tr = Tree.fromstring(item)
		# lstRules.append(tr.productions())
		lstRules = lstRules + tr.productions()


	# make rules unique(no duplicates)
	if not isDuplicate:
		return list(set(lstRules))
	else:
		return lstRules


###################################################################################
# write rules for "devset.trees" to a file (for easier copy-paste/editing)
def writeRulesToFile(filename, isDups=False):
	if isDups:
		filename=filename+"WITHDUPS"
	else:
		filename=filename+"WITHOUTDUPS"
	rules = makeRulesFromTreeList(getTreesFromDevset("devset.trees"), isDuplicate=isDups)
	with open(filename, "w") as file:
		for rule in rules:
			line = "1" + "    " + str(rule) + '\n'
			file.write(line)
###################################################################################

# Separating rules and terminals from the devset.trees
# returns a tuple of lists (terminal list and rules list)
# NOTE: rules is still not in CNF
def getRulesAndTerminals(strFilename, isWriteToFile=False):
	terminals=[]
	rules=[]
	with open(strFilename) as file:
		lines = file.readlines()
		for currentline in lines:
			currentline = currentline.partition("\n")[0]
			if currentline[-1] == "'":
				terminals.append(currentline)
			else:
				rules.append(currentline)


	terminals = list(set(terminals))
	rules = list(set(rules))
	# sort to be easier to read
	terminals.sort()
	rules.sort()

	# write to file for easier copy-paste/editing
	file_devset_terminals="DevSetTerminals.txt"
	file_devset_rules="DevSetRules_NONCNF.txt"
	if isWriteToFile:
		with open(file_devset_terminals, "w") as file:
			for item in terminals:
				line = item + '\n'
				file.write(line)

		with open(file_devset_rules, "w") as file:
			for item in rules:
				line = item + '\n'
				file.write(line)


	return (terminals, rules)

