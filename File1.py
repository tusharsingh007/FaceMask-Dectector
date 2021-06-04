#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install opencv-python')


# In[35]:


import cv2


# In[36]:


img = cv2.imread('Tushar.jpg')


# In[37]:


img.shape


# In[38]:


img[0]


# In[39]:


import matplotlib.pyplot as plt


# In[40]:


while True:
    cv2.imshow('result',img)
    if cv2.waitKey(2) == 27:
        break
cv2.destroyAllWindows()


# In[41]:


haar_data = cv2.CascadeClassifier('data.xml')


# In[42]:


haar_data.detectMultiScale(img)


# In[43]:


while True:
    faces = haar_data.detectMultiScale(img)
    for x,y,w,h in faces:
        cv2.rectangle(img,(x,y) , (x+w, y+h) , (255,0,255) , 4)
    cv2.imshow('result',img)
    #27-ASCII of Escape
    if cv2.waitKey(2) == 27:
        break
cv2.destroyAllWindows()


# In[50]:


plt.imshow(img)


# In[49]:


capture = cv2.VideoCapture(0)
data = []
while True:
    flag, img = capture.read()
    if flag:
        faces = haar_data.detectMultiScale(img)
        for x,y,w,h in faces:
            cv2.rectangle(img,(x,y) , (x+w, y+h) , (255,0,255) , 4)
            face = img[y:y+h , x:x+w, :] 
            face = cv2.resize(face,(50,50))
            print(len(data))
            if len(data) < 400:
                data.append(face) 
        cv2.imshow('result', img)
        if cv2.waitKey(2) == 27 or len(data) >= 200:
            break
capture.release()
cv2.destroyAllWindows()


# In[51]:


import numpy as np


# In[52]:


x = np.array([3,4,5])


# In[53]:


x


# In[54]:


x = np.array([[4,45,7] , [7,8,9] , [8,9,6]])


# In[55]:


x


# In[56]:


x[0]


# In[57]:


x[0][1:2]


# In[58]:


x[0:2 , 1:2 ]


# In[32]:


np.save('withouut_mask.npy', data)


# In[34]:


np.save('with_mask.npy', data)


# In[ ]:




