import pandas as pd
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

#implementing linear regression
#assuming a kernel to be a [1]
# close_form_solution
def LR(data,y):
    w=np.dot(np.dot(inv(np.dot(data.T,data)),data.T),y)
    return w

# gradient_descent_solution

def LR_grad(data,y,alpha):
    w=np.zeros((data.shape[1],1))
    
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
        msk = np.random.rand(len(mnist_data)) < 0.7
        train=data[msk]
        y_train=y[msk]
        test=data[~msk]
        y_test=y[msk]
    elif(arg==3):
        data=pd.read_csv("fmnist-pca.csv")
        data=np.array(data)
        y=data[:,0]
        data=data[:,1:]
        msk = np.random.rand(len(mnist_data)) < 0.7
        train=data[msk]
        y_train=y[msk]
        test=data[~msk]
        y_test=y[msk]
    elif(arg==4):
        data=pd.read_csv("Assignment2Data_4.csv")
        data=np.array(data)
        y=data[:,1]
        data=data[:,0]
        msk = np.random.rand(len(data)) < 0.7
        train=data[msk]
        y_train=y[msk]
        test=data[~msk]
        y_test=y[msk]
        
    return train.reshape((train.shape[0],1)),test.reshape((test.shape[0],1)),y_train.reshape((y_train.shape[0],1)),y_test.reshape((y_test.shape[0],1))
 
 
def dimension(data,dim):
    for i in range(2,dim+1):
        col=np.power(data[:,1],i).reshape(data.shape[0],1)
        data=np.hstack((data,col))
    return data
        
        
        
train1,test1,y_train,y_test=file_handler(2)
b=np.ones((train1.shape[0],1))
train1=np.hstack((b,train1))
c=np.ones((test1.shape[0],1))
test1=np.hstack((c,test1))

dim=8
plt.figure()
# plt.scatter(data1[:,0],y)
error=[]
dim_=[]
for i in range(2,dim):
    print(i)
    data=dimension(train1,i)
    test_1=dimension(test1,i)
    w=LR(data,y_train)
    w=np.array(w)
    res=np.dot(test_1,w)
    plt.scatter(test_1[:,1],res[:,0],label="i")
    plt.title('Linear Regression')
    plt.ylabel('Oxygen Levels')
    plt.xlabel('x')
    plt.ylim(0,100)
    plt.grid(b=True)
    count=0
    for j in range(test_1.shape[0]):
        count+=((res[j]-y_test[j])*(res[j]-y_test[j]))     
    error_avg=count/data.shape[0]
    error.append(error_avg)
    dim_.append(i)
    print(i,error_avg)
plt.show()
plt.plot(dim_,error,'r')
plt.title("Least Sqaure Error VS dimesnsion")
plt.ylabel("Least Sqaure Error")
plt.xlabel("dimension")
plt.grid(b=True)
plt.show()