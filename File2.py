#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


import cv2


# In[4]:


with_mask = np.load('with_mask.npy')


# In[7]:


without_mask = np.load('withouut_mask.npy')


# In[9]:


with_mask.shape


# In[10]:


without_mask.shape


# In[11]:


with_mask = with_mask.reshape(200,50 * 50 * 3)
without_mask = without_mask.reshape(200,50 * 50 * 3)


# In[12]:


with_mask.shape


# In[13]:


without_mask.shape


# In[14]:


X = np.r_[with_mask, without_mask]


# In[15]:


X.shape


# In[16]:


labels = np.zeros(X.shape[0])


# In[40]:


labels[200:] = 1.0
names = {0 : 'Mask', 1 : 'NO Mask'}


# In[41]:


#svm = Support vector machine
#SVC - Support vector classification
from sklearn.svm import SVC 
from sklearn.metrics import accuracy_score


# In[42]:


from sklearn.model_selection import train_test_split


# In[61]:


x_train, x_test, y_train, y_test = train_test_split(X,labels, test_size=0.25)


# In[62]:


x_train.shape


# In[45]:


#PCA - principal component analysis
from sklearn.decomposition import PCA


# In[84]:


# 3 - 3D data
pca = PCA(n_components = 3) 
x_train = pca.fit_transform(x_train)


# In[85]:


x_train[0]


# In[86]:


x_train.shape


# In[87]:


x_train, x_test, y_train, y_test = train_test_split(X,labels, test_size=0.25)


# In[88]:


svm = SVC()
svm.fit(x_train, y_train)


# In[89]:


#x_test = pca.transform(x_test)
y_pred = svm.predict(x_test)


# In[90]:


accuracy_score(y_test, y_pred)


# In[91]:


haar_data = cv2.CascadeClassifier('data.xml')
capture = cv2.VideoCapture(0)
data = []
font = cv2.FONT_HERSHEY_COMPLEX
while True:
    flag, img = capture.read()
    if flag:
        faces = haar_data.detectMultiScale(img)
        for x,y,w,h in faces:
            cv2.rectangle(img,(x,y) , (x+w, y+h) , (255,0,255) , 4)
            face = img[y:y+h , x:x+w, :] 
            face = cv2.resize(face,(50,50))
            face = face.reshape(1,-1)
            # face = pca.transform(face)
            pred= svm.predict(face) 
            n = names[int(pred)]
            cv2.putText(img, n, (x,y), font, 1, (244,250,255), 2)
            print(n)
        cv2.imshow('result', img)
        if cv2.waitKey(2) == 27:
            break
capture.release()
cv2.destroyAllWindows()


# In[ ]:




