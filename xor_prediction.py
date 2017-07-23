
# coding: utf-8

# In[1]:


from keras.models import Sequential
from keras.layers import Dense
import numpy


# In[2]:


# training datas
train = numpy.array([
    [0, 0, 0],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0],
     [0, 0, 0],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0],
     [0, 0, 0],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0],
     [0, 0, 0],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0],
     [0, 0, 0],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0],
     [0, 0, 0],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0]
])
inputX = train[:, 0:2]
inputY = train[:, 2]


# In[3]:


print(inputX)
print(inputY)


# In[4]:


# The sequential model is a linear stack of layers
model = Sequential()
model.add(Dense(2, input_dim=2))
model.add(Dense(3, activation='sigmoid'))
model.add(Dense(1))


# In[5]:


model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])


# In[6]:


model.fit(inputX, inputY, epochs=1000, batch_size=4)


# In[7]:


scores = model.evaluate(inputX, inputY)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# In[8]:


prediction = model.predict(numpy.array([[0,0], [0, 1], [1, 0], [0, 0]]))


# In[9]:


print(prediction)


# In[11]:





# In[ ]:




