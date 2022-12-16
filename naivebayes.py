#Naive Bayes Approach
import numpy as np
import math
from sklearn.metrics import precision_score, recall_score, f1_score

def separate_by_class(x):
	separated = dict()
	for i in range(len(x)):
		row = list(x[i])            
		cv = row.pop(-1)            # pop class to store as key
		if (cv not in separated):
			separated[cv] = list()
		separated[cv].append(row)
	return separated


def summarise(seperated):
	summary = dict()
	for label,value in seperated.items():
		summary[label] = [(np.mean(c), np.std(c), len(c)) for c in zip(*value)]
	return summary

#predicting each class for given row
# def calcCprob(summaries, row):
# 	total_rows = sum([summaries[label][0][2] for label in summaries])
# 	probabilities = dict()
# 	for cv, cs in summaries.items():
# 		#calculating prior for the class
# 		probabilities[cv] = summaries[cv][0][2]/float(total_rows)
# 		for i in range(len(cs)):
# 			mean, std, _ = cs[i]
# 			probabilities[cv] *= (1 / (np.sqrt(2 * math.pi) * std)) * np.exp(-((row[i]-mean)**2 / (2 * std**2 )))
# 	return probabilities

#With log trick
def calcCprob(summaries, row):
	total_rows = sum([summaries[label][0][2] for label in summaries])
	probabilities = dict()
	for cv, cs in summaries.items():
		#calculating prior for the class
		probabilities[cv] = np.log(summaries[cv][0][2]/float(total_rows))
		for i in range(len(cs)):
			mean, std, _ = cs[i]
			if std>0.2:
				prob = (1 / (np.sqrt(2 * math.pi) * std)) * np.exp(-((row[i]-mean)**2 / (2 * std**2 )))
			else:
				prob = 0
			if prob != 0:
				probabilities[cv] += np.log(prob)
	return probabilities

def predict(summaries, row):
	probabilities = calcCprob(summaries, row)
	bestLabel, bestProb = None, -1
	for cv, prob in probabilities.items():
		if bestLabel is None or prob > bestProb:
			bestProb = prob
			bestLabel = cv
	return bestLabel
 
def accuracy(y,yhat):
	ate = 0
	for i in range(yhat.shape[0]):
		if y[i,:] == yhat[i,:]:
			ate += 1
	ate = ate*100/yhat.shape[0]
	return ate


np.random.seed(0)

#LOADING DATA
X = np.loadtxt(open("data.csv", "rb"), delimiter=",", skiprows=1)[:,1:]
Y = np.loadtxt(open("data.csv", "rb"), delimiter=",", skiprows=1)[:,0]
Y = Y.reshape(Y.shape[0],1)

#Z-SCORE
meanX = np.mean(X, axis=0)
stdX = np.std(X, ddof=1, axis=0)
stdX[stdX==0] = 1
zscored = (X - meanX) / stdX
X = zscored
#print(X.shape,Y.shape)

X = np.concatenate((X,Y),axis=1)
#-------------------------------------------------------------------------------------------------------

#SHUFFLE
p = np.random.permutation(len(X))
X_train, X_test = X[p][:math.ceil(2*len(X)/3),:], X[p][math.ceil(2*len(X)/3):,:]
# print(X_train.shape, X_test.shape)

sep = separate_by_class(X_train)
model = summarise(sep)
#print(len(model[0.0]))
#print(model)


#validation into x and y
valx = X_test[:,:-1]
valy = X_test[:,-1]
valy = valy.reshape(valy.shape[0],1)

#print(valx.shape,valy.shape)

yhat = []

for row in valx:
	y_hat = predict(model, row)
	yhat.append(y_hat)

yhat = np.array(yhat)
yhat = yhat.reshape(valy.shape)

#note: to run the code with log trick please uncomment the function above. 
print(f'Precision: {precision_score(valy, yhat)}')
print(f'Recall: {recall_score(valy,yhat)}')
print(f'F-measure: {f1_score(valy,yhat)}')
print(f'Accuracy: {accuracy(valy,yhat)} %')

