import numpy as np
from scipy.stats import entropy

def get_entropy(X,eps=5.0):  # <eps则归为一个数据  # X: array
    # 合并同类项：
    X=sort(X)
    n=len(X)
    n_ = np.float(n)
    f=0.0
    prob=[]
    rules=X[0]
    for i in range(n):
        if X[i]-eps<=rules<=X[i]+eps :
            f+=1.0
        else:

            p=f/n_
            prob.append(f/n_)
            f=0
            rules=X[i].copy()
            f+=1
    prob.append(f/n_)
    prob=np.array(prob).squeeze()
    h=entropy(prob)
    return h



def sort(arr):

        n = len(arr)

        # 遍历所有数组元素
        for i in range(n):

            # Last i elements are already in place
            for j in range(0, n - i - 1):

                if arr[j] > arr[j + 1]:
                    arr[j], arr[j + 1] = arr[j + 1].copy(), arr[j].copy()
        return arr