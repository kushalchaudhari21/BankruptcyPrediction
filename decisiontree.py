#Decision Tree approach
import numpy as np
import pandas as pd
import math
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from collections import Counter
from tqdm import tqdm

class Node:
	def __init__(self, features=None, threshold=None, lt=None, rt=None, gain=None, value=None):
		self.features = features
		self.threshold = threshold
		self.lt = lt
		self.rt = rt
		self.gain = gain
		self.value = value

class Classifier:
	#splits and depths
	def __init__(self, sp=2, d=5):
		self.sp = sp
		self.d = d
		self.root = None
   
	#entropy
	def ent(self,colList):
	 
		# no. of rows in a category for feature
		value_counts = np.bincount(np.array(colList, dtype=np.int64))
		# Prob of each class
		percentages = value_counts / len(colList)

		#ent
		ent = 0
		for p in percentages:
			if p > 0:
				ent += p * np.log2(p)
		return -ent
	
	#information gain
	def ig(self, parent, lc, rc):
	
		lcn = len(lc) / len(parent)
		rcn = len(rc) / len(parent)
		return self.ent(parent) - (lcn * self.ent(lc) + rcn * self.ent(rc))
	
	def bs(self, x, y):
		
		bs = {}
		bIG = -1
		rows, cols = x.shape
		for idx in tqdm(range(cols)):
			X_f = x[:, idx]
			
			# For every unique value in feature
			for threshold in np.unique(X_f):
				
				# dataset right part greater thatn threshold, left less than equal to. 
				df = np.concatenate((x, y.reshape(1, -1).T), axis=1)
				df_left = np.array([row for row in df if row[idx] <= threshold])
				df_right = np.array([row for row in df if row[idx] > threshold])
				if len(df_left) > 0 and len(df_right) > 0:
					# get target label vals for all ele left and right of dataset
					y = df[:, -1]
					y_left = df_left[:, -1]
					y_right = df_right[:, -1]

					# Caclulate ig on y target, y left target and y right target
					# if the current split is better than bIG, then we update the variable bIG
					gain = self.ig(y, y_left, y_right)
					if gain > bIG:
						bs = {'feature_index': idx,'threshold': threshold,'df_left': df_left,'df_right': df_right,'gain': gain}
						bIG = gain
		return bs
	
	
	def build(self, x, y, d=0):
  
		rows, cols = x.shape
		
		if rows >= self.sp and d <= self.d: # in this case our node is not a leaf
			# We call the bs function defined above to get the best split for the gain
			best = self.bs(x, y)
		
			if best['gain'] > 0:
				# Build a tree on the left
				left = self.build(
					x=best['df_left'][:, :-1], 
					y=best['df_left'][:, -1], 
					d=d + 1
				)
				
				# Build a tree on the right
				right = self.build(
					x=best['df_right'][:, :-1], 
					y=best['df_right'][:, -1], 
					d=d + 1
				)
				
				# return the node with the left and right subtree
				return Node(
					features=best['feature_index'], 
					threshold=best['threshold'], 
					lt=left, 
					rt=right, 
					gain=best['gain']
				)
		# If the node is a leaf -  return the most common target value 
		return Node(
			value=Counter(y).most_common(1)[0][0]
		)
	
	def fit(self, x, y):
		
		# Call a recursive function to build the tree
		self.root = self.build(x, y)
		
	def subp(self, x, tree):
	  
		# Leaf node
		if tree.value != None:
			return tree.value
		feature = x[tree.features]
		
		# Left subtree
		if feature <= tree.threshold:
			return self.subp(x=x, tree=tree.lt)
		
		# Right Subtree
		if feature > tree.threshold:
			return self.subp(x=x, tree=tree.rt)
		
	def predict(self, X):
		
		# Call the subp() recursively function for every observation
		return [self.subp(x, self.root) for x in X]
	
def accuracy(y,yhat):
	ate = 0
	for i in range(len(yhat)):
		if y[i] == yhat[i]:
			ate += 1
	ate = ate*100/len(yhat)
	return ate


if __name__=="__main__":
	np.random.seed(0)
	datafile = './data.csv'
	df = pd.read_csv(datafile,delimiter=',')
	df = df.drop(index = 0)

	feature = df.drop(columns=["Bankrupt?"], axis=1)
	#dropping features
	feature = feature[feature.columns[-25:]].copy()
	label = df.iloc[:,0]

	dataStd = (feature - feature.mean())/feature.std()

	X_train, X_test, y_train, y_test = train_test_split(dataStd, label, train_size=math.ceil(2*dataStd.shape[0]/3))
	model = Classifier()
	model.fit(np.array(X_train), np.array(y_train))
	yhat = model.predict(np.array(X_test))
	y_test = np.array(y_test)

	print(f'Accuracy: {accuracy(y_test,yhat)} %')

