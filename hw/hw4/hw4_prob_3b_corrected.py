import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as Lin_Reg

# Fit
def ridge(x_train, y_train, reg_param):
    n=np.shape(x_train)[0]
    x_train=np.concatenate((x_train,np.sqrt(reg_param)*np.identity(n)))
    y_train_=np.zeros((n+np.shape(x_train)[1],1))
    for c in range(n):
        y_train_[c]= y_train[c]
    import sklearn
    model = sklearn.linear_model.LinearRegression()
    model.fit(x_train,y_train_)
    return model

# Score
def score(m,x_test,y_test, reg_param):
    n=np.shape(x_train)[0]
    x_test=np.concatenate((x_test,np.sqrt(reg_param)*np.identity(n)))
    y_test_=np.zeros((n+np.shape(x_test)[1],1))
    for c in range(n):
        y_test_[c]= y_test[c]
    return m.score(x_test,y_test_)

# Load
data = np.loadtxt('datasets/dataset_3.txt', delimiter=',')
n = data.shape[0]
n = int(np.round(n*0.5))
x_train = data[0:n,0:100]
y_train = data[0:n,100]
x_test = data[n:2*n,0:100]
y_test = data[n:2*n,100]

# Params
a=np.zeros(5)
for i in range(-2,3):
    a[i+2]=10**i

# Iterate
rstr =np.zeros(5)
rsts =np.zeros(5)
for j in range(0,5):
    m =ridge(x_train,y_train,a[j])
    rstr[j]=score(m,x_train,y_train,a[j])
    rsts[j]=score(m,x_test,y_test,a[j])

# Plot
plt.plot(a,rstr)
plt.plot(a,rsts)

plt.xscale('log')


plt.show()
