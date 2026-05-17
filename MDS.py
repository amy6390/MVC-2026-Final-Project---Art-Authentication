from sklearn.manifold import MDS
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(678)

def fitto3d(vectors):
    mds=MDS(n_components=3, random_state=11242, dissimilarity='euclidean')
    coords=mds.fit_transform(vectors)
    return coords

def display_wavelets():
    vangogh_vectors=[]
    with open("Real Painting Features.txt", 'r') as f:
        for i in range(8):
            v=[float(j) for j in f.readline().split(' ')]
            vangogh_vectors.append(v)

    AI_vectors=[]
    with open('AI Painting Features.txt','r') as f:
        for i in range(7):
            v=[float(j) for j in f.readline().split(' ')]
            AI_vectors.append(v)

    all_vectors=vangogh_vectors+AI_vectors
    coords=fitto3d(all_vectors)
    print(coords)

    #plotting coords in 3d space
    fig=plt.figure()
    ax=fig.add_subplot(projection='3d')
    vgx, vgy, vgz=[coords[i][0] for i in range(8)], [coords[i][1] for i in range(8)], [coords[i][2] for i in range(8)]
    aix, aiy, aiz=[coords[i][0] for i in range(8, 15)], [coords[i][1] for i in range(8, 15)], [coords[i][2] for i in range(8, 15)]

    ax.scatter(vgx, vgy, vgz, color='red', label='Van Gogh')
    ax.scatter(aix, aiy, aiz, color='blue', label='AI')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.legend()
    plt.show()

def display_curvature():
    vangogh_vectors=[]
    with open("Curvature - Real Painting Features.txt", 'r') as f:
        for i in range(8):
            v=[float(j) for j in f.readline().split(' ')]
            vangogh_vectors.append(v)

    AI_vectors=[]
    with open('Curvature - AI Painting Features.txt','r') as f:
        for i in range(7):
            v=[float(j) for j in f.readline().split(' ')]
            AI_vectors.append(v)

    all_vectors=vangogh_vectors+AI_vectors
    coords=fitto3d(all_vectors)
    print(coords)

    #plotting coords in 3d space
    fig=plt.figure()
    ax=fig.add_subplot(projection='3d')
    vgx, vgy, vgz=[coords[i][0] for i in range(8)], [coords[i][1] for i in range(8)], [coords[i][2] for i in range(8)]
    aix, aiy, aiz=[coords[i][0] for i in range(8, 15)], [coords[i][1] for i in range(8, 15)], [coords[i][2] for i in range(8, 15)]

    ax.scatter(vgx, vgy, vgz, color='red', label='Van Gogh')
    ax.scatter(aix, aiy, aiz, color='blue', label='AI')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.legend()
    plt.show()
    #plt.savefig('MDS of Wavelet Analysis.jpeg') (saved manually)

display_wavelets()
display_curvature()
