from sklearn.cross_decomposition import PLSRegression
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
from sklearn import kernel_approximation as ks
from sklearn import preprocessing

class KernelPLS(PLSRegression):
    #this class is an extention of PLS which uses kernels . we call it kernel pls now!
    

    def __init__(self ,copy=True, max_iter=500, n_components=1, scale=True, tol=1e-06,kernel = 'rbf',preprocess = True):
        super(KernelPLS, self).__init__(copy=copy, max_iter=max_iter, n_components=n_components, scale=scale, tol=tol)
        self.kernel = kernel
        self.preprocess = preprocess
    def convert_to_kernel(self,X0):
        kX = ks.Nystroem(kernel=self.kernel,degree=1)
        Xkernel = kX.fit_transform(X0)
        if self.preprocess:
            Xscaler  = preprocessing.StandardScaler().fit(Xkernel)
            Xkernel = Xscaler.transform(Xkernel)
            self.Xscaler = Xscaler
        #print(Xkernel)
        self.Xkernel = Xkernel
        self.X0 = X0
        self.kX = kX
        return Xkernel
    def convert_to_kernel_pred(self,X):
        kX = self.kX
        Xkernel = kX.transform(X)
        if self.preprocess:
            Xscaler = self.Xscaler
            Xkernel = Xscaler.transform(Xkernel)
        return Xkernel
        
    def fit(self,X,Y):
        z = super(KernelPLS,self).fit(X = self.convert_to_kernel(X), Y = y) 
        return z
    def predict(self,X , copy = True ):
        z = super(KernelPLS,self).predict(X = self.convert_to_kernel_pred(X), copy = copy)
        return z
        

