import numpy as np
from cv2 import *
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt 
data = np.genfromtxt('finaltraining.csv',delimiter = ',' )
xtrain = data[:,:3]

#print(xtrain)
xtarget = data[:,3:]
#print(xtarget)
img1 = imread('m_image/cm.jpg')
img = cvtColor(img1,COLOR_BGR2RGB)
m,n,p = img.shape
#print(n)
R = np.reshape(img[:,:,0],(m*n,1));
G = np.reshape(img[:,:,1],(m*n,1));
B = np.reshape(img[:,:,2],(m*n,1));
xtest = np.hstack([R,G,B])
clf = MLPClassifier(solver='lbfgs', alpha=0.7, hidden_layer_sizes=(100, 2),activation='logistic')
print(clf.fit(xtrain,xtarget))
out = clf.predict(xtest)
out_img = np.reshape(out,(m,n))
print(out_img)
imgr = np.zeros((m,n,3))
for i in range(0,m):
	for j in range(0,n):
		if out_img[i,j] == 2:
			imgr[i,j,:] = [255,0,0]
		else:
			imgr[i,j,:] = img[i,j,:]
		#print(g+1)
plt.subplot(211)
plt.imshow(img)
plt.subplot(212)
plt.imshow(imgr)
plt.show()
