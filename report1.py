import numpy as np
import matplotlib.pyplot as plt
import time

def train(w,b,t):
    w_ = w.copy()
    for i in range(100000):
        t_=t[np.random.randint(len(t))][:,np.newaxis]
        w_ = i/(i+1)*w_ + 1/(i+1)*np.dot(t_,t_.T)
        w = w*(1-np.identity(25))
        w = (w+w.T)/2
    return w_

def test(w,t,p):
    plt.figure(figsize=(1,1))
    for j in range(len(t)):
        x_old = t[j][:,np.newaxis]*((np.random.randint(0,100,(25,1))>(p-1))*2-1)
        for i in range(100):
            ind = np.random.randint(25)
            x_new = x_old.copy()
            x_new[ind] = (np.dot(w[ind:ind+1,:],x_old)>=0)*2-1
            x_old = x_new.copy()
            if i%5==0:
                plt.imshow(x_new.reshape(5,5),"gray")
                plt.pause(0.1)
if __name__=="__main__":
    t = np.array([[1,1,-1,1,1,1,-1,1,-1,1,1,-1,1,-1,1,1,-1,1,-1,1,1,1,-1,1,1],
                  [-1,1,1,1,1,-1,1,1,1,1,-1,1,1,1,1,-1,1,1,1,1,-1,1,1,1,1],
                  [1,1,-1,-1,-1,1,1,1,1,-1,1,1,-1,-1,-1,1,1,-1,1,1,1,1,-1,-1,-1],
                  [-1,-1,-1,1,1,1,1,-1,1,1,1,-1,-1,1,1,1,1,-1,1,1,-1,-1,-1,1,1],
                  [1,-1,1,-1,1,1,-1,1,-1,1,1,-1,-1,-1,1,1,1,1,-1,1,1,1,1,-1,1],
                  [-1,-1,-1,-1,-1,-1,1,1,1,1,-1,-1,-1,-1,-1,1,1,1,1,-1,-1,-1,-1,-1,-1]])
    w = np.random.uniform(-1,1,(25,25))
    b = np.ones((25,1))
    w = train(w,b,t)
    test(w,t,10)



