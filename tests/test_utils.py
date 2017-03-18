from unittest import TestCase
from ntsa.utils import *
from ntsa.data_generator import create_data
import matplotlib.pyplot as plt
import pickle
import time

# Read Lorenz data
with open('lorenz_data', 'rb') as file:
    content = pickle.load(file)
data = content['data'].T  # Lorenz x-component time series
ts = content['ts']  # sampling interval


class TestUtils(TestCase):
    def test_entropy(self):
        print('\nTest: entropy...')
        x = np.random.rand(1000)
        counts, bins = np.histogram(x, bins=100)
        px = counts / np.sum(counts)
        h = entropy(px)
        self.failUnless(h >= 0)
        print('Entropy: {}'.format(h))
        print('ok...')

    def test_mutual_information(self):
        print('\nTest: mutual information...')

        # Test for random data
        x = np.random.randn(1000)
        y = np.random.randn(1000)
        var1 = x + y
        var2 = x - y
        mi = mutual_information(var1, var2)
        self.failUnless(mi >= 0)
        print('Mutual information: {}'.format(mi))

        mi = np.array([])
        for l in range(1, 50):
            a = data[:-l]
            b = data[l:]
            mi = np.append(mi, mutual_information(a, b))
        plt.stem(np.arange(1, 50), mi)
        plt.xlabel('lag')
        plt.ylabel('I(lag)')
        plt.show()
        time.sleep(5)
        print('ok...')

    def test_optimum_delay(self):
        print('\nTest: optimum delay...')
        with open('lorenz_data', 'rb') as file:
            content = pickle.load(file)
        data = content['data'].T
        l = optimum_delay(data)
        print('optimum lag: {}'.format(l))
        print('ok..')

    def test_delay_embedding(self):
        print('\nTest: Delay embedding...')
        # delay with d = 1
        data = np.arange(6)
        embedded = delay_embedding(data, 4)
        expected = np.array([[0, 1, 2, 3],
                             [1, 2, 3, 4],
                             [2, 3, 4, 5]])
        self.failUnless(np.allclose(embedded, expected))

        # delay with d > 1
        data = np.arange(10)
        embedded = delay_embedding(data, 4, lag=2)
        expected = np.array([[0, 2, 4, 6],
                             [1, 3, 5, 7],
                             [2, 4, 6, 8],
                             [3, 5, 7, 9]])
        self.failUnless(np.allclose(embedded, expected))
        print('ok..')

        # delay with d > 1
        data = np.arange(10)
        embedded = delay_embedding(data, 4, lag=2)
        expected = np.array([[0, 2, 4, 6],
                             [1, 3, 5, 7],
                             [2, 4, 6, 8],
                             [3, 5, 7, 9]])
        self.failUnless(np.allclose(embedded, expected))
        print('ok..')

    def test_nearest_neighbor(self):
        print('\nTest: k nearest neighbour...')
        data = np.array([[1, 1],
                         [2, 2],
                         [5, 5],
                         [6, 6]])
        indexes, neighbors = nearest_neighbors(data, neigh=2, get_neighbors=True)
        self.failUnless(np.allclose(indexes, np.array([1, 0, 3, 2])))
        self.failUnless(np.allclose(neighbors, np.array([[2., 2.],
                                                         [1., 1.],
                                                         [6., 6.],
                                                         [5., 5.]])))
        print('ok..')

    def test_embedding_dimension(self):
        print('\nTest: embedding_dimension...')
        l = optimum_delay(data)
        embed_dim, percentages = embedding_dimension(data, l, max_dim=10, get_fnn=True)
        plt.stem(np.arange(1, 10), percentages)
        plt.xlabel('Dimension')
        plt.ylabel('False nearest neighbor ratio')
        plt.show()
        print('ok...')

    def test_lyapunov(self):
        print('\nTest: Lyapunov exponent...')
        lyap_exp, y = lyapunov(data, ts, lag=11, dim=3, num_states=300, get_divergence=True)
        t = np.arange(len(y)) * ts
        div_calculated = y * ts
        div_expected = 1.5 * t
        plt.plot(t, div_calculated, label='Calculated')
        plt.plot(t, div_expected, '--', label='Theoretical')
        plt.xlabel('Time')
        plt.ylabel('<ln(divergence)>')
        plt.title('lyap_exp: {}'.format(lyap_exp))
        plt.legend()
        plt.show()
        print('ok...')

    def test_optimum_lag(self):
        print('\nTest: optimum lag')
        n = 5
        data = np.arange(n)
        """
        For this data auto-correlation is equal to np.array([30., 20., 11., 4., 0.])
        Optimum delay d at which auto-correlation value is (1 - 1/np.e) * 30 = 18.963 => d = 2
        """
        d = optimum_delay(data)
        self.failUnless(d == 2)

        # test for the lorenz time series
        with open('lorenz_data', 'rb') as file:
            content = pickle.load(file)
        data = content['data']
        d = optimum_delay(data)
        print(d)
        """
        Rosenstein et.al estimates the optimum delay as 11 for 50000 sample lorenz time series
        obtained by Runga-Kutta integration with a step size of 0.01
        """

        # try for different lorenz time series
        for i in range(10):
            data = create_data('lorenz_parametric', np.random.rand(3), 'lorenz_data', out=True)
            d = optimum_delay(data)
            print(d)
        print('ok...')
