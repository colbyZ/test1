import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as Lin_Reg


# Fit
def ridge(x_train, y_train, reg_param):
    # copy-and-paste code at the beginning of ridge() and score() methods
    # len(x_train) is shorter
    n=np.shape(x_train)[0]
    # concatenation should be done with regards to rows (axis=0)
    x_train=np.concatenate((x_train,reg_param*np.identity(n)),axis=1)
    # 'y_train_' is a bad name since it can be easily confused with 'y_train'
    # this code could written as: 'np.concatenate((y, np.zeros(x_train.shape[1]))).reshape(-1, 1)'
    # 'y_train_' is unused
    y_train_=np.zeros((n+np.shape(x_train)[1],1))
    for c in range(n):
        y_train_[c]= y_train[c]
    # imports usually are at the beginning of the code
    import sklearn
    model = sklearn.linear_model.LinearRegression()
    # no need to reshape; looks like 'y_train_' is supposed to be used here
    model.fit(x_train,y_train.reshape(-1,1))
    return model

# Score
def score(m,x_test,y_test, reg_param):
    # copy-and-paste code from ridge() with the same problems
    n=np.shape(x_train)[0]
    x_test=np.concatenate((x_test,reg_param*np.identity(n)),axis=1)
    y_test_=np.zeros((n+np.shape(x_test)[1],1))
    for c in range(n):
        y_test_[c]= y_test[c]
    return m.score(x_test,y_test.reshape(-1,1))

# Load
data = np.loadtxt('datasets/dataset_3.txt', delimiter=',')
# len(data) is simpler
n = data.shape[0]
# n // 2 is simpler
n = int(np.round(n*0.5))
# data[:n, :-1]
x_train = data[0:n,0:100]
# data[:n, -1]
y_train = data[0:n,100]
# data[n:, :-1]
x_test = data[n:2*n,0:100]
# data[n:, -1]
y_test = data[n:2*n,100]

# Params
# 'alphas' could be a better name instead of 'a'
# this could written as 'alphas = [10.0 ** i for i in range(-2, 3)]'
a=np.zeros(5)
# the last element of 'a' is 0.0 instead of expected '10 ** 2'
# should be 'range(-2, 3)'
for i in range(-2,2):
    a[i+2]=10**i

# Iterate
# 'rstr' and 'rsts' are cryptic names; better names: 'train_scores', 'test_scores'
rstr =np.zeros(5)
rsts =np.zeros(5)
for j in range(0,5):
    # should be 'a[j]'
    m =ridge(x_train,y_train,a[i])
    rstr[j]=score(m,x_train,y_train,a[j])
    # should be 'rsts[j]', 'a[j]'
    rsts[i]=score(m,x_test,y_test,a[i])

# Plot
# x scale should be 'log'
# missing in the plot: xlabel, ylabel, title, legend
# without them and without any context it's not clear what the plot is about
plt.plot(a,rstr)
plt.plot(a,rsts)


plt.show()
