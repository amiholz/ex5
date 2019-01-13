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
    return reverse_eig_vec*np.sqrt(reverse_eig_val)


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

def display_and_save_with_labels(data, labels, title, save=True):
    plt.figure()
    MDS_title = "MDS - MNIST"
    plt.title(MDS_title, fontsize=20)
    plt.scatter(X_MDS[:,0], X_MDS[:,1], c=digits_labels, cmap="gist_rainbow")
    plt.savefig(MDS_title)
    plt.show()

def MNIST(digits_data, digits_labels):

    # Distance matrix and dimension
    d = 2
    X = euclidean_distances(digits_data, squared=True)

    # PART I - MDS
    X_MDS = MDS(X, d)
    display_and_save_with_labels(X_MDS, digits_labels, "MDS - MNIST")

    # PART II - LLE
    K = [5, 10, 50, 100]
    X_LLE = []
    for k in K:
        result = LLE(digits_data, d, k)
        X_LLE.append(result)

    # PART III - DM
    sigma = [0.5,2,10]
    T = [2,20]
    X_DM = []
    for t in T:
        for s in sigma:
            result = DiffusionMap(digits_data, d, s, t)
            X_DM.append(result)



def Swiss_Roll(swiss_roll_data, swiss_roll_labels):
    pass

def Faces(faces_data):
    pass

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

    MNIST(digits_data, digits_labels)
    # Swiss_Roll(swiss_roll_data, swiss_roll_labels)
    # Faces(faces_data)