import numpy as np


class RandomData:
    """
    A class defining a dataset genrated using a random procedure for the features as well as the labels.
    """
    dev = 0.1

    def __init__(self, k=3, r=1., n=1000, d=20):
        """
        Initializing the class from passed arguments
        :param k: int, k^2 = number of clusters
        :param r: float, bound on the data : it will ly in [-r,r]^d
        :param n: int, total number of samples to generate (to be later divided into train, val, test)
        :param d: int, dimension of the data to be generated
        """
        if k <= 0:
            raise ValueError("k argument must be > 0 but was {:,}".format(k))
        if r <= 0:
            raise ValueError("r argument must be > 0 but was {:.5f}".format(r))
        if d < 2:
            raise ValueError("d argument must be >= 2 but was {:,}".format(d))
        self.k = k
        self.r = r
        self.n = n
        self.d = d
        self.B = 2 * r / (3 * k - 1)

        self.line_coordinates_ = []
        self.cluster_centers_ = np.array([])
        self.cluster_labels_ = np.array([])
        self.data_points_ = np.zeros((self.n, self.d))
        self.data_points_cluster_assignments_ = np.zeros(self.n)
        self.data_points_labels_ = np.ones(self.n)

    def generate_samples(self, n: [int, None] = None):
        """
        First sample k^2 cluster centers, and assign a label to each possible cluster uniformly over {-1, 1}. Then, for
        each data point, sample a cluster center uniformly and then sample a data point uniformly in an l2 ball of
        radius B = 2r / (3k - 1) around the cluster center. Points from any 2 different clusters should be at distance
        at least B of each other.
        :return:
        """
        if n is None:
            n = self.n  # set n to be the value used when the class was initialized
        self.sample_cluster_centers()
        self.sample_cluster_labels()
        self.data_points_cluster_assignments_ = np.random.choice(np.arange(len(self.cluster_centers)), size=n)
        uniform_samples_in_ball = self.uniform_sample_ball_(n)
        data_points_centers = np.array([self.cluster_centers[self.data_points_cluster_assignments_[i]]
                                        for i in range(n)])
        self.data_points_labels_ = np.array([self.cluster_labels[self.data_points_cluster_assignments_[i]]
                                             for i in range(n)])
        self.data_points_ = data_points_centers + uniform_samples_in_ball

        return self.data_points_, self.data_points_cluster_assignments_, self.data_points_labels_

    @property
    def line_coordinates(self):
        if len(self.line_coordinates_) == 0:
            # coordinates of cluster centers in the same line are spread on the line [-r, r] with a step of 3B.
            self.line_coordinates_ = [-self.r + self.B + 3 * l * self.B for l in range(self.k)]
        return self.line_coordinates_

    @property
    def cluster_centers(self):
        if len(self.cluster_centers_) == 0:
            self.sample_cluster_centers()
        return self.cluster_centers_

    @property
    def cluster_labels(self):
        if len(self.cluster_labels_) == 0:
            self.sample_cluster_labels()
        return self.cluster_labels_

    @property
    def data_points(self):
        return self.data_points_

    @property
    def data_points_labels(self):
        return self.data_points_labels_

    def sample_cluster_centers(self):
        """
        Sample k^2 possible cluster centers. For each cluster center, each coordinate in R^d is drawn uniformly from the
        possible line coordinates.
        :return:
        """
        # when sampling randomly, the same cluster center might appear twice, so we keep sampling until we reach the
        # desired length
        cluster_centers = set()
        while len(cluster_centers) < self.k ** 2:
            cluster_centers.add(tuple(np.random.choice(self.line_coordinates, size=self.d)))
        self.cluster_centers_ = np.array(list(cluster_centers))

    def sample_cluster_labels(self):
        """
        There are k^2 clusters. Each one is assigned a random label uniformly over {-1, 1}.
        :return:
        """
        cluster_labels = np.random.choice([-1, 1], size=self.k ** 2)
        positive_ratio = np.sum((cluster_labels + 1) / 2) / len(cluster_labels)
        # make sure the cluster labels are balanced enough
        cmpt = 0
        while (cmpt < 100) and ((positive_ratio < 0.5 - self.dev) or (positive_ratio > 0.5 + self.dev)):
            cluster_labels = np.random.choice([-1, 1], size=self.k ** 2)
            positive_ratio = np.sum((cluster_labels + 1) / 2) / len(cluster_labels)
            cmpt += 1

        if cmpt == 100:
            raise ValueError("cmpt = 100 which means max retries have been reached when trying to sample balanced "
                             "cluster labels")
        self.cluster_labels_ = cluster_labels

    def uniform_sample_ball_(self, n) -> np.array:
        """
        Samples and returns n points uniformly over a ball of radius B centered at 0 in R^d. The method consists in
        first sampling gaussian variable in dimension d with identity covariance matrix, then normalizing the vector,
        and finally mutliplying by a number drawn uniformly from [0,1] and taken to the power 1/d. See
        http://extremelearning.com.au/how-to-generate-uniformly-random-points-on-n-spheres-and-n-balls/ for more
        details.
        :return: np.array with n samples drawn uniformly from the ball of radius B in dimension d.
        """
        u = np.random.normal(0, 1, size=(n, self.d))
        norm = np.sum(u ** 2, axis=1) ** 0.5  # size n
        r = np.random.uniform(0, 1, size=n) ** (1.0 / self.d)
        # (r * u / norm) is drawn uniformly from a ball of radius 1
        return self.B * u * (r / norm).reshape(n, 1)
