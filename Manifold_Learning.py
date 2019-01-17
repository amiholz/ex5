import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap, LocallyLinearEmbedding
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import pdist, squareform

save = True

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

def eig_decomposition(mat, descending=False):
    """
    Decompose the given matrix to its eigenvalues and eigenvectors
    and return them sorted

    :param mat: size (N, N) input matrix to decompose
    :param descending: set to True for descending order (largest to smallest)
    :return: (N,) array of sorted eigenvalues, and their (N, N) eigen vectors
    """
    eig_values, eig_vectors = np.linalg.eigh(mat)
    sorted_indices = eig_values.argsort()
    if descending is True:
        sorted_indices = np.flip(sorted_indices, axis=0)
    return eig_values[sorted_indices], eig_vectors[:, sorted_indices]


def MDS(X, d):
    '''
    Given a NxN pairwise distance matrix and the number of desired dimensions,
    return the dimensionally reduced data points matrix after using MDS.

    :param X: NxN distance matrix.
    :param d: the dimension.
    :return: Nxd reduced data point matrix.
    '''
    N = X.shape[0]
    H = np.eye(N) - np.ones((N,N))/N
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
    knn = np.argsort(euclidean_distances(X), axis=1)[:,1:k+1]
    W = np.zeros((knn.shape[0], knn.shape[0]))
    for i in range(W.shape[0]):
        Z = X[knn[i]]
        Z = Z-X[i]
        G = np.inner(Z,Z)
        w = np.linalg.pinv(G).dot(np.ones(G.shape[0]))
        W[i,knn[i]] = w/np.sum(w)

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

def points_display(data, labels, title, save=True, cmap = "gist_rainbow"):
    plt.figure()
    plt.title(title, fontsize=20)
    plt.scatter(data[:,0], data[:,1], c=labels, cmap=cmap)
    if save: plt.savefig(title)
    plt.show()



def MNIST(digits_data, digits_labels):
    # Distance matrix and dimension
    d = 2
    X = euclidean_distances(digits_data, squared=True)

    # PART I - MDS
    X_MDS = MDS(X, d)[0]
    points_display(X_MDS, digits_labels, "MNIST - MDS", save)

    # PART II - LLE
    K = [5, 10, 20, 50]
    LLE_title = "MNIST - LLE"
    plt.figure()
    plt.suptitle(LLE_title)
    for k in range(len(K)):
        data = LLE(digits_data, d, K[k])
        plt.subplot(1,len(K),k+1)
        plt.title("k="+str(K[k]) , fontsize=15)
        axis=np.min(data[:,0]),np.max(data[:,0]),np.min(data[:,1]),np.max(data[:,1])
        plt.axis(axis)
        plt.scatter(data[:, 0], data[:, 1], c=digits_labels, cmap="gist_rainbow")
    if save: plt.savefig(LLE_title)
    plt.show()

    # PART III - DM
    T = [1, 20, 50]
    DM_title = "MNIST - Diffusion Maps"
    plt.figure()
    plt.suptitle(DM_title, fontsize=15)
    for t in range(len(T)):
        data = DiffusionMap(digits_data, d, 10, T[t])
        plt.subplot(1, len(T),t+1)
        plt.title("t="+str(T[t])+" S=50" , fontsize=10)
        axis=np.min(data[:,0]),np.max(data[:,0]),np.min(data[:,1]),np.max(data[:,1])
        plt.axis(axis)
        plt.scatter(data[:, 0], data[:, 1], c=digits_labels, cmap="gist_rainbow")
    if save: plt.savefig(DM_title)
    plt.show()

def Swiss_Roll(swiss_roll_data, swiss_roll_labels):

    # Distance matrix and dimension
    d = 2
    X = euclidean_distances(swiss_roll_data, squared=True)

    # PART I - MDS
    X_MDS = MDS(X, d)[0]
    points_display(X_MDS, swiss_roll_labels, "Swiss Rol - MDS", save, "inferno")

    # PART II - LLE
    K = [5, 10, 30, 50, 100]
    LLE_title = "Swiss Rol - LLE"
    plt.figure()
    plt.suptitle(LLE_title)
    for k in range(len(K)):
        data = LLE(swiss_roll_data, d, K[k])
        plt.subplot(1,len(K),k+1)
        plt.title("k="+str(K[k]) , fontsize=15)
        axis=np.min(data[:,0]),np.max(data[:,0]),np.min(data[:,1]),np.max(data[:,1])
        plt.axis(axis)
        plt.scatter(data[:, 0], data[:, 1], c=swiss_roll_labels, cmap="inferno")
    if save: plt.savefig(LLE_title)
    plt.show()

    # PART III - DM
    S = [1,5,9]
    DM_title = "Swiss Rol - Diffusion Maps"
    plt.figure()
    for s in range(len(S)):
        data = DiffusionMap(swiss_roll_data, d, S[s], 2)
        plt.subplot(1,len(S),s+1)
        plt.title("T=2"+" S="+str(S[s]) , fontsize=10)
        axis=np.min(data[:,0]),np.max(data[:,0]),np.min(data[:,1]),np.max(data[:,1])
        plt.axis(axis)
        plt.scatter(data[:, 0], data[:, 1], c=swiss_roll_labels, cmap="inferno")
    if save: plt.savefig(DM_title)
    plt.show()

def Faces(faces_data):
    save = True
    # Distance matrix and dimension
    d = 2
    X = euclidean_distances(faces_data, squared=True)

    # PART I - MDS
    plot_with_images(MDS(X, d)[0], faces_data, "Faces - MDS", 70)
    plt.show()

    # PART II - LLE
    K = [5, 50, 100]
    for k in range(len(K)):
        plot_with_images(LLE(X, d, K[k]), faces_data, "Faces - LLE", 70)
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

def random_rotation_matrix(d, loc=0.0, scale=0.1):
    '''
    Generate a uniformly random rotation matrix

    :param loc: mean of the distribution.
    :param scale: std of the distribution.
    :param d: the dimension.
    :return:
    '''
    rnd = np.random.normal(loc=loc, scale=scale, size=[d, d])
    q, _ = np.linalg.qr(rnd)

    return q

def scree(sd = 1):
    # mu, sigma = (3, 5), np.array([[40, -6], [-6, 3]])
    # noise_sigma_values = [0.5, 2, 5, 10]
    # high_dim = 10
    # n_samples = 100
    # # create Gaussian data with some fixed mean and variance
    # # and embed it in higher dimension
    # data_2d = np.random.multivariate_normal(mu, sigma, size=n_samples).T
    # rotation_mat = np.random.normal(size=(high_dim, 2))
    # rotation_mat, _ = np.linalg.qr(rotation_mat)
    # data_embedded = (np.matmul(rotation_mat,data_2d)).T


    n = 50
    intrinsic_d = 2
    ambient_d = 5
    data = np.random.random([n, intrinsic_d]) * 10
    # extend to ambient_d dimensions
    data_extended = np.append(data, np.zeros([n, ambient_d - intrinsic_d]), axis=1)
    # randomly rotate
    rot = random_rotation_matrix(ambient_d)
    rotated_data = np.dot(data_extended, rot)

    # try different noises
    for i, scale in enumerate([0.1, 0.2, 0.5, 1, 1.5, 2, 4, 7, 10]):
        plt.subplot(3, 3, i+1)
        plt.title('noise std=' + str(scale))
        plt.xlabel('eigenvalue id')
        plt.ylabel('eigenvalue')
        # add gaussian noise
        noisy_data = rotated_data + np.random.normal(scale=scale, size=rotated_data.shape)
        distances = squareform(pdist(noisy_data))
        _, w = MDS(distances, intrinsic_d)
        plt.scatter(range(len(w)), w)

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
    Swiss_Roll(swiss_roll_data, swiss_roll_labels)
    # Faces(faces_data)
    # scree(0)

    # pca = PCA(n_components=2)
    # new = pca.fit_transform(digits_data)
    # plt.figure()
    # plt.scatter(new[:,0], new[:,1], c=digits_labels, cmap="gist_rainbow")
    # plt.show()
    exit(0)

    isomap = LocallyLinearEmbedding(n_components=2, n_neighbors=5)
    new = isomap.fit_transform(swiss_roll_data)
    plt.figure()
    plt.scatter(new[:,0], new[:,1], c=swiss_roll_labels, cmap="viridis")
    plt.show()
    isomap = LocallyLinearEmbedding(n_components=2, n_neighbors=15)
    new = isomap.fit_transform(swiss_roll_data)
    plt.figure()
    plt.scatter(new[:,0], new[:,1], c=swiss_roll_labels, cmap="plasma")
    plt.show()
    isomap = LocallyLinearEmbedding(n_components=2, n_neighbors=50)
    new = isomap.fit_transform(swiss_roll_data)
    plt.figure()
    plt.scatter(new[:,0], new[:,1], c=swiss_roll_labels, cmap="inferno")
    plt.show()
    isomap = LocallyLinearEmbedding(n_components=2, n_neighbors=100)
    new = isomap.fit_transform(swiss_roll_data)
    plt.figure()
    plt.scatter(new[:,0], new[:,1], c=swiss_roll_labels, cmap="magma")
    plt.show()
    exit(0)
    #
    #
