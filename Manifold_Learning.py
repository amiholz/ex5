import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.neighbors import kneighbors_graph
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial import distance_matrix

def digits_example():
    '''
    Example code to show you how to load the MNIST data and plot it.
    '''

    # load the MNIST data:
    digits = datasets.load_digits()
    data = digits.data / 255.
    labels = digits.target

    # plot examples:
    plt.gray()
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.axis('off')
        plt.imshow(np.reshape(data[i, :], (8, 8)))
        plt.title("Digit " + str(labels[i]))
    plt.show()

def swiss_roll_example():
    '''
    Example code to show you how to load the swiss roll data and plot it.
    '''

    # load the dataset:
    X, color = datasets.samples_generator.make_swiss_roll(n_samples=5000)

    # plot the data:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)
    plt.show()


def faces_example(path):
    '''
    Example code to show you how to load the faces data.
    '''

    with open("faces.pickle", 'rb') as f:
        X = pickle.load(f)

    num_images, num_pixels = np.shape(X)
    d = int(num_pixels**0.5)
    print("The number of images in the data set is " + str(num_images))
    print("The image size is " + str(d) + " by " + str(d))

    # plot some examples of faces:
    plt.gray()
    for i in range(4):
        plt.subplot(2, 2, i+1)
        plt.imshow(np.reshape(X[i, :], (d, d)))
    plt.show()


def plot_with_images(X, images, title, image_num=25):
    '''
    A plot function for viewing images in their embedded locations. The
    function receives the embedding (X) and the original images (images) and
    plots the images along with the embeddings.

    :param X: Nxd embedding matrix (after dimensionality reduction).
    :param images: NxD original data matrix of images.
    :param title: The title of the plot.
    :param num_to_plot: Number of images to plot along with the scatter plot.
    :return: the figure object.
    '''

    n, pixels = np.shape(images)
    img_size = int(pixels**0.5)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(title)

    # get the size of the embedded images for plotting:
    x_size = (max(X[:, 0]) - min(X[:, 0])) * 0.08
    y_size = (max(X[:, 1]) - min(X[:, 1])) * 0.08

    # draw random images and plot them in their relevant place:
    for i in range(image_num):
        img_num = np.random.choice(n)
        x0, y0 = X[img_num, 0] - x_size / 2., X[img_num, 1] - y_size / 2.
        x1, y1 = X[img_num, 0] + x_size / 2., X[img_num, 1] + y_size / 2.
        img = images[img_num, :].reshape(img_size, img_size)
        ax.imshow(img, aspect='auto', cmap=plt.cm.gray, zorder=100000,
                  extent=(x0, x1, y0, y1))

    # draw the scatter plot of the embedded data points:
    ax.scatter(X[:, 0], X[:, 1], marker='.', alpha=0.7)

    return fig

def MDS(X, d):
    '''
    Given a NxN pairwise distance matrix and the number of desired dimensions,
    return the dimensionally reduced data points matrix after using MDS.

    :param X: NxN distance matrix.
    :param d: the dimension.
    :return: Nxd reduced data point matrix.
    '''
    N = X.shape[0]
    H = np.eye(N) - (1/N)*np.ones((N,N))
    S = -0.5*np.matmul(H, np.matmul(X, H))
    eig_val, eig_vec= np.linalg.eigh(S)
    reverse_eig_val = np.flip(eig_val, axis=0)[:d]
    reverse_eig_vec = np.flip(eig_vec, axis=1)[:,:d]
    return reverse_eig_vec*np.sqrt(reverse_eig_val), eig_val


def LLE(X, d, k):
    '''
    Given a NxD data matrix, return the dimensionally reduced data matrix after
    using the LLE algorithm.

    :param X: NxD data matrix.
    :param d: the dimension.
    :param k: the number of neighbors for the weight extraction.
    :return: Nxd reduced data matrix.
    '''
    knn = kneighbors_graph(X, k).toarray()
    W = np.zeros((knn.shape))
    for i in range(W.shape[0]):
        Z = X[np.where(knn[i]==1)[0]]
        Z = Z-X[i]
        G = np.inner(Z,Z)
        w = np.linalg.pinv(G).dot(np.ones(G.shape[0]))
        W[i,np.where(knn[i]==1)] = w/np.sum(w)

    M = np.eye(W.shape[0]) - W
    eig_val, eig_vec= np.linalg.eigh(np.matmul(M.T,M))
    return eig_vec[:,1:d+1]

def DiffusionMap(X, d, sigma, t):
    '''
    Given a NxD data matrix, return the dimensionally reduced data matrix after
    using the Diffusion Map algorithm. The k parameter allows restricting the
    kernel matrix to only the k nearest neighbor of each data point.

    :param X: NxD data matrix.
    :param d: the dimension.
    :param sigma: the sigma of the gaussian for the kernel matrix transformation.
    :param t: the scale of the diffusion (amount of time steps).
    :return: Nxd reduced data matrix.
    '''
    K = np.exp(-(euclidean_distances(X, X)**2)/(sigma))
    A = K/K.sum(axis=1)[:, np.newaxis]
    eig_val, eig_vec= np.linalg.eigh(A)
    reverse_eig_val = np.flip(eig_val, axis=0)[1:d+1]
    reverse_eig_vec = np.flip(eig_vec,axis=1)[:,1:d+1]
    return reverse_eig_vec*np.power(reverse_eig_val, t)

def points_display(data, labels, title, save=True):
    plt.figure()
    plt.title(title, fontsize=20)
    plt.scatter(data[:,0], data[:,1], c=labels, cmap="gist_rainbow")
    if save: plt.savefig(title)
    plt.show()

def MNIST(digits_data, digits_labels):

    save = True
    # Distance matrix and dimension
    d = 2
    X = euclidean_distances(digits_data, squared=True)

    # PART I - MDS
    X_MDS = MDS(X, d)[0]
    points_display(X_MDS, digits_labels, "MNIST - MDS", save)

    # PART II - LLE
    K = [5, 50, 100, 200]
    LLE_title = "MNIST - LLE"
    plt.figure()
    plt.suptitle(LLE_title)
    for k in range(len(K)):
        # data1 = LLE1(digits_data, d, K[k])
        data = LLE(digits_data, d, K[k])
        plt.subplot(1,len(K),k+1)
        plt.title("k="+str(K[k]) , fontsize=15)
        plt.scatter(data[:, 0], data[:, 1], c=digits_labels, cmap="gist_rainbow")
    if save: plt.savefig(LLE_title)
    plt.show()

    # PART III - DM
    S = [0.5,2,10]
    T = [2,20]
    DM_title = "MNIST - Diffusion Maps"
    plt.figure()
    plt.suptitle(DM_title, fontsize=15)
    count=1
    for t in range(len(T)):
        for s in range(len(S)):
            data = DiffusionMap(digits_data, d, S[s], T[t])
            plt.subplot(len(T),len(S),count)
            count+=1
            plt.title("t="+str(T[t])+" S="+str(S[s]) , fontsize=10)
            plt.scatter(data[:, 0], data[:, 1], c=digits_labels, cmap="gist_rainbow")
    if save: plt.savefig(DM_title)
    plt.show()

def Swiss_Roll(swiss_roll_data, swiss_roll_labels):

    save = True
    # Distance matrix and dimension
    d = 2
    X = euclidean_distances(swiss_roll_data, squared=True)

    # PART I - MDS
    X_MDS = MDS(X, d)[0]
    points_display(X_MDS, swiss_roll_labels, "Swiss Rol - MDS", save)

    # PART II - LLE
    K = [5, 50, 100]
    LLE_title = "Swiss Rol - LLE"
    plt.figure()
    plt.suptitle(LLE_title)
    for k in range(len(K)):
        # data1 = LLE1(digits_data, d, K[k])
        data = LLE(swiss_roll_data, d, K[k])
        plt.subplot(1,len(K),k+1)
        plt.title("k="+str(K[k]) , fontsize=15)
        plt.scatter(data[:, 0], data[:, 1], c=swiss_roll_labels, cmap="gist_rainbow")
    if save: plt.savefig(LLE_title)
    plt.show()

    # PART III - DM
    S = [0.5,7]
    T = [2,20]
    DM_title = "Swiss Rol - Diffusion Maps"
    plt.figure()
    plt.suptitle(DM_title, fontsize=15)
    count=1
    for t in range(len(T)):
        for s in range(len(S)):
            data = DiffusionMap(swiss_roll_data, d, S[s], T[t])
            plt.subplot(len(T),len(S),count)
            count+=1
            plt.title("t="+str(T[t])+" S="+str(S[s]) , fontsize=10)
            plt.scatter(data[:, 0], data[:, 1], c=swiss_roll_labels, cmap="gist_rainbow")
    if save: plt.savefig(DM_title)
    plt.show()

def Faces(faces_data):
    save = True
    # Distance matrix and dimension
    d = 2
    X = euclidean_distances(faces_data, squared=True)

    # PART I - MDS
    plot_with_images(MDS(X, d)[0], faces_data, "Faces - MDS", 50)

    # PART II - LLE
    K = [5, 50, 100]
    for k in range(len(K)):
        plot_with_images(LLE(X, d, K[k]), faces_data, "Faces - LLE", 50)
    plt.gray()
    plt.show()

    # PART III - DM
    S = [40,70]
    T = [2,20]
    for t in range(len(T)):
        for s in range(len(S)):
            plot_with_images(DiffusionMap(X, d, S[s], T[t]), faces_data, "Faces - MDS", 50)
    plt.gray()
    plt.show()

def scree(sd = 1):
    mu, sigma = (3, 5), np.array([[40, -6], [-6, 3]])
    noise_sigma_values = [0.5, 2, 5, 10]
    high_dim = 10
    n_samples = 100
    # create Gaussian data with some fixed mean and variance
    # and embed it in higher dimension
    data_2d = np_random.multivariate_normal(mu, sigma, size=n_samples).T
    rotation_mat = np_random.normal(size=(high_dim, 2))
    rotation_mat, _ = np.linalg.qr(rotation_mat)
    data_embedded = (rotation_mat @ data_2d).T


    D=10
    N=500
    data2D = np.zeros((N, D))
    data2D[:2,:] = np.random.normal(size=(2, D))

    Q = np.linalg.qr(np.random.normal(size=(N, N)))[0]
    dataHD = np.matmul(Q,data2D)

    dataHDnoise = np.random.normal(size=(N, D), scale=sd)

    data = dataHD + dataHDnoise

    X = euclidean_distances(data, squared=True)

    eig_val = MDS(X, 2)[1][1:]

    plt.plot(np.arange(len(eig_val)), eig_val)
    plt.show()

if __name__ == '__main__':
    '''
    Preparation of the data
    '''
    # DIGITS
    digits = datasets.load_digits()
    digits_data = digits.data / 255.
    digits_labels = digits.target
    # SWISS_ROLL
    swiss_roll_data, swiss_roll_labels = datasets.samples_generator.make_swiss_roll(n_samples=5000)
    # FACES
    # with open("faces.pickle", 'rb') as f:
    #     faces_data = pickle.load(f)

    # MNIST(digits_data, digits_labels)
    # Swiss_Roll(swiss_roll_data, swiss_roll_labels)
    # Faces(faces_data)
    scree(0)