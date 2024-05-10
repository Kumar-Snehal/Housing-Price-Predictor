from math import nan
import numpy as np
from matplotlib import pyplot as plt

def CostFunc(x,y,w,b):
    l=x.shape[0]
    cost=0
    for i in range(l):
        Y=w*x[i]+b
        cost+=((Y-y[i])**2)/(2*l)
    return cost

def calcGrad(x,y,w,b):
    m = x.shape[0]    
    dJdw = 0
    dJdb = 0
    for i in range(m):  
        Y = w*x[i]+b
        dJdb += (Y-y[i])/m
        dJdw += ((Y-y[i])/m)*x[i]
    return dJdw, dJdb

def GradDes(x,y,w,b):
    iterations=10000
    learnRate=0.1
    while(iterations):
        prevcost=CostFunc(x,y,w,b)
        wstep,bstep=calcGrad(x,y,w,b)
        w=w-learnRate*wstep
        b=b-learnRate*bstep
        cost=CostFunc(x,y,w,b)
        if(cost>prevcost):
            w+=wstep
            b+=bstep
            break
        iterations-=1
    return w,b        

def FeatureScale(x,y):
    xfac=x[0]
    yfac=y[0]
    for i in range(x.shape[0]):
        if(x[i]>xfac):
            xfac=x[i]
        if(y[i]>yfac):
            yfac=y[i]
    return xfac,yfac

def linearReg(x,y):
    w=0
    b=0
    xS,yS=FeatureScale(x,y)
    X=x/xS
    Y=y/yS
    w,b=GradDes(X,Y,w,b)
    return w,b,xS,yS
    
n=int(input("Enter number of training values: "))
x=np.arange(n,dtype=float)
y=np.arange(n,dtype=float)
for i in range(n):
    print("Enter ",i+1,"th training set [House Size, Price]: ",end="")
    a,b=map(float,input().split())
    x[i]=a
    y[i]=b
print(x)
print(y)
plt.plot(x,y)
plt.show()
w,b,xS,yS=linearReg(x,y)
Y=np.arange(n,dtype=float)
for i in range(n):
    Y[i]=w*(x[i]/xS)+b
Y=Y*yS
print(Y)
print("w=",w,"\nb=",b)
plt.plot(x,Y)
plt.show()
print("Enter House Size: ")
check=float(input())
outcheck=(w*(check/xS)+b)*yS
print("The Price should be ",outcheck)