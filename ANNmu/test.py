from cv2 import *
import matplotlib.pyplot as plt
import numpy as np
img1 = imread('m_image/bm2.jpg')
img = cvtColor(img1,COLOR_BGR2RGB)
m,n,p = img.shape
rows, cols = (m, n) 
arr = [[0]*cols]*rows
R = np.reshape(img[:,:,0],(m*n,1));
G = np.reshape(img[:,:,1],(m*n,1));
B = np.reshape(img[:,:,2],(m*n,1));
c = np.reshape(arr,(m*n,1))
tot = np.hstack([R,G,B,c])
#print(tot)
np.savetxt('training4.csv',tot,delimiter=',',fmt='%d')
'''plt.subplot(211)
plt.imshow(img)
plt.show()'''