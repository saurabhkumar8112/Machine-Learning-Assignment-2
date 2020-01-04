import pandas as pd
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

#implementing linear regression
#assuming a kernel to be a [1]
# close_form_solution


def sigmoid(data):
    return (1/(1+np.exp(-data)))

def log_likelihood(data,y,w):
    z=np.dot(data,w)
    print(z,np.log(1+np.exp(z)))
    ll=np.sum(y*z-np.log(1+np.exp(z)))
    return ll

    
def logistic(data,y,iteration,alpha):
    w=np.zeros((data.shape[1],1))
    print(np.dot(data,w))
    for i in range(iteration):
        z=np.dot(data,w)
        predict=np.zeros((data.shape[0],1))
        sig=sigmoid(z)
        for j in range(data.shape[0]):
            if(sig[j]>=0.5):
                predict[j]=1
            else:
                print(predict[j])
        error=y-predict
        grad=np.dot(data.T,error)
        w+=(alpha*grad)
        if(i%1000==0):
            print(log_likelihood(data,y,w))
    return w
# data=pd.read_csv("railwayBookingList.csv").replace(replacements)
def file_handler(arg):
    if(arg==1):
        replacements={
        "MEDICATION":0,
        "HEALTHY":1,
        "SURGERY":2
        }  
        data=pd.read_csv("Medical_data.csv").replace(replacements)
        data=np.array(data)
        y=data[:,0]
        data=data[:,1:]
        print(data.shape)
    elif(arg==2):
        replacements = {
            'FIRST_AC': 1,
            'SECOND_AC': 2,
            'THIRD_AC': 3,
            'NO_PREF': 4,
            'male' : 1,
            'female' : 0
        }
        data=pd.read_csv("railwayBookingList.csv").replace(replacements)
        data=np.array(data)
        y=data[:,1]
        y=[(y[a]-0.5)*2 for a in range(data.shape[0])]
        data=data[:,2:]
    elif(arg==3):
        data=pd.read_csv("fmnist-pca.csv")
        data=np.array(data)
        y=data[:,0]
        data=data[:,1:]
    elif(arg==4):
        data=pd.read_csv("Assignment2Data_4.csv")
        data=np.array(data)
        y=data[:,1]
        data=data[:,0]
    return data.reshape((data.shape[0],1)),y.reshape((y.shape[0],1))
 
 
def dimension(data,dim):
    for i in range(2,dim+1):
        col=np.power(data[:,1],i).reshape(data.shape[0],1)
        data=np.hstack((data,col))
    return data
        
        
        
data1,y=file_handler(4)
b=np.ones((data1.shape[0],1))
data1=np.hstack((b,data1))
logistic(data1,y,30000,0.05)
'''dim=10
plt.figure()
plt.scatter(data1[:, 1],y)
error=[]
dim_=[]'''
'''for i in range(2,dim):
    print(i)
    data=dimension(data1,i)
    w=LR(data,y)
    w=np.array(w)
    res=np.dot(data,w)
    plt.scatter(data[:,1],res[:,0],label="i")
    plt.title('Linear Regression')
    plt.ylabel('Oxygen Levels')
    plt.xlabel('x')
    plt.ylim(0,100)
    plt.grid(b=True)
    count=0
    for j in range(data.shape[0]):
        count+=((res[j]-y[j])*(res[j]-y[j]))     
    error_avg=count/data.shape[0]
    error.append(error_avg)
    dim_.append(i)
    print(i,error_avg)
plt.show()
plt.plot(dim_,error,'r')
plt.title("Least Sqaure Error VS dimesnsion")
plt.ylabel("Least Sqaure Error")
plt.xlabel("dimension")
plt.grid(b=True)'''
plt.show()