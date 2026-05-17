import numpy as np
import matplotlib.pyplot as plt
import pywt
from sklearn import linear_model
import scipy
import pandas

def grayscale(img_path):
    vals=plt.imread(img_path)
    print("shape of file is:", vals.shape)
    gray_vals=np.dot(vals[...,:3], [0.299, 0.587, 0.114]) #converts values to grayscale using dot product (gray = 0.229R + 0.587G + 0.114B)

    plt.imshow(gray_vals, cmap='gray')
    plt.title("Grayscale Painting")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

    print("gray scale image created")
    return gray_vals

def wavelets(grayscale_vals):
    coeffs=pywt.dwt2(grayscale_vals, 'db1')

    cA, (cH, cV, cD)=coeffs

    plt.imshow(cA, cmap='gray')
    plt.title("Approximation")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

    plt.imshow(cH, cmap='gray')
    plt.title("Horizontal Detail")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

    plt.imshow(cV, cmap='gray')
    plt.title("Vertical Detail")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

    plt.imshow(cD, cmap='gray')
    plt.title("Diagonal Detail")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

    return coeffs

def extract_stats(arr): #input should be 2D numpy array
    #get mean, variance, skewness, from data set
    mean=np.mean(arr)
    variation=np.std(arr)**2
    skewness=scipy.stats.skew(arr, axis=None)
    kurtosis=scipy.stats.kurtosis(arr, axis=None)

    x_dir=[-1, 0, 1]
    y_dir=[-1, 0, 1]
    N, M=arr.shape[0], arr.shape[1]
    X=[] #inputs to LinReg model
    y=[] #actual outputs for model to fit to
    #create data frame with values from neighbors of current points
    for i in range(1, N-1):
        for j in range(1, M-1):
            curr_neighbors=[]
            for cx in x_dir:
                for cy in y_dir:
                    if cx==0 and cy==0:
                        continue
                    curr_neighbors.append(arr[i+cx][j+cy])
            X.append(curr_neighbors) #neighbors
            y.append(arr[i][j]) #actual value
    
    print('Data for linear regression created')
    #run prediction algorithm from scipy and find error
    model=linear_model.LinearRegression()
    model.fit(X, y)
    print('Model created')
    print("R-squared:", model.score(X, y))
    predicted=model.predict(X)
    errors=abs(predicted-np.array(y))
    error_mean, error_variance, error_skewness, error_kurtosis=np.mean(errors), np.std(errors)**2, scipy.stats.skew(errors, axis=None), scipy.stats.kurtosis(errors, axis=None)

    return (mean, variation, skewness, kurtosis, error_mean, error_variance, error_skewness, error_kurtosis)

file_names=['Skull', 'Starry Night', 'The Irises', 'Burning Cigarette', 'Sunflowers', 'The Potato Eaters', "Van Gogh's Chair", 'Cafe Terrace']
with open("Real Painting Features.txt", 'w') as f:
    for name_ in file_names:
        gray_vals=grayscale(name_+".jpg")
        cA, (cH, cV, cD)=wavelets(gray_vals)
        feature_vector=[]

        vals=extract_stats(cH)
        for i in vals:
            feature_vector.append(i)
        vals=extract_stats(cV)
        for i in vals:
            feature_vector.append(i)
        vals=extract_stats(cD)
        for i in vals:
            feature_vector.append(i)
        
        #print(feature_vector)
        f.write(' '.join([str(i) for i in feature_vector]))
        f.write('\n')

file_names=['AI Skull', 'AI Starry Night', 'AI Irises', 'AI Skeleton', 'AI Cafe', 'AI House', 'AI Sunflower']
with open("AI Painting Features.txt", 'w') as f:
    for name_ in file_names:
        gray_vals=grayscale(name_+".jpg")
        cA, (cH, cV, cD)=wavelets(gray_vals)
        feature_vector=[]

        vals=extract_stats(cH)
        for i in vals:
            feature_vector.append(i)
        vals=extract_stats(cV)
        for i in vals:
            feature_vector.append(i)
        vals=extract_stats(cD)
        for i in vals:
            feature_vector.append(i)
        
        #print(feature_vector)
        f.write(' '.join([str(i) for i in feature_vector]))
        f.write('\n')