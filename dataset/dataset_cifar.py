from dataset import *
import cPickle
import numpy as np


class CifarDataset(Dataset):
    def __init__(self):
        Dataset.__init__(self)
        self.data_dims = [32, 32, 3]
        self.range = [0.0, 1.0]
        self.batch_size = 100
        self.name = "cifar"
        self.folder = "cifar"
        self.train_data = np.zeros((50000, 32, 32, 3))
        for i in range(5):
            data, labels = self.unpickle(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                      self.folder, 'data_batch_%d' % (i+1)))
            self.train_data[i*10000:(i+1)*10000] = np.transpose(np.reshape(data, (-1, 3, 32, 32)), (0, 2, 3, 1))
        data, labels = self.unpickle(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                  self.folder, 'test_batch'))
        self.test_data = np.transpose(np.reshape(data, (-1, 3, 32, 32)), (0, 2, 3, 1))
        self.train_batch_ptr = 0
        self.test_batch_ptr = 0
        self.train_data = np.clip(self.train_data / 255.0, a_min=0.0, a_max=1.0)
        self.test_data = np.clip(self.test_data / 255.0, a_min=0.0, a_max=1.0)
        self.train_size = self.train_data.shape[0]
        self.test_size = self.test_data.shape[0]

    @staticmethod
    def unpickle(filename):
        fo = open(filename, 'rb')
        content = cPickle.load(fo)
        fo.close()
        return content['data'], content['labels']

    def next_batch(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        prev_batch_ptr = self.train_batch_ptr
        self.train_batch_ptr += batch_size
        if self.train_batch_ptr > self.train_data.shape[0]:
            self.train_batch_ptr = batch_size
            prev_batch_ptr = 0
        result = self.train_data[prev_batch_ptr:self.train_batch_ptr, :, :, :]
        return result

    def next_test_batch(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        prev_batch_ptr = self.test_batch_ptr
        self.test_batch_ptr += batch_size
        if self.test_batch_ptr >self.test_data.shape[0]:
            self.test_batch_ptr = batch_size
            prev_batch_ptr = 0
        result = self.test_data[prev_batch_ptr:self.test_batch_ptr, :, :, :]
        return result

    def batch_by_index(self, batch_start, batch_end):
        return self.train_data[batch_start:batch_end, :, :, :]

    def display(self, image):
        return np.clip(image, 0.0, 1.0)

    def reset(self):
        self.train_batch_ptr = 0
        self.test_batch_ptr = 0

if __name__ == '__main__':
    dataset = CifarDataset()
    images = dataset.next_batch()
    for i in range(100):
        plt.imshow(dataset.display(images[i]))
        plt.show()