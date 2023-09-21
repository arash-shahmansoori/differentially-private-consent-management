import numpy as np
import random
import torch
import torchvision.transforms as transforms

# Training context window size
NUM_WIN_SIZE = 200  # 200ms == 2 seconds
SHORT_SIZE = 100  # 100ms == 1 seconds

SAMPLE_RATE = 16000
FILTER_BANK = 40


class ToTensorInput(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, np_feature):
        """
        Args:
            feature (numpy.ndarray): feature to be converted to tensor.
        Returns:
            Tensor: Converted feature.
        """
        if isinstance(np_feature, np.ndarray):
            # handle numpy array
            ten_feature = torch.from_numpy(
                np_feature.transpose((0, 2, 1))
            ).float()  # output type => torch.FloatTensor, fast

            # input size : (1, n_win=200, dim=40)
            # output size : (1, dim=40, n_win=200)
            return ten_feature


class TruncatedInputfromMFB(object):
    """
    input size : (n_frames, dim=40)
    output size : (1, n_win=40, dim=40) => one context window is chosen randomly
    """

    def __init__(self, input_per_file=1):
        super(TruncatedInputfromMFB, self).__init__()
        self.input_per_file = input_per_file

    def __call__(self, frames_features):
        # Normalizing before slicing
        network_inputs = []
        num_frames = len(frames_features)
        win_size = NUM_WIN_SIZE
        half_win_size = int(win_size / 2)
        # if num_frames - half_win_size < half_win_size:
        while num_frames <= win_size:
            frames_features = np.append(
                frames_features, frames_features[:num_frames, :], axis=0
            )
            num_frames = len(frames_features)

        for i in range(self.input_per_file):
            j = random.randrange(half_win_size, num_frames - half_win_size)
            if not j:
                frames_slice = np.zeros(num_frames, FILTER_BANK, "float64")
                frames_slice[0 : (frames_features.shape)[0]] = frames_features.shape
            else:
                frames_slice = frames_features[j - half_win_size : j + half_win_size]
            network_inputs.append(frames_slice)
        return np.array(network_inputs)


class metaGenerator(object):
    def __init__(
        self,
        data_DB,
        file_loader,
        nb_classes=40,
        nb_samples_per_class=3,
        max_iter=50,
        xp=np,
    ):
        super(metaGenerator, self).__init__()

        self.nb_classes = nb_classes
        self.nb_samples_per_class = nb_samples_per_class
        self.max_iter = max_iter
        self.xp = xp
        self.num_iter = 0
        self.data = self._load_data(data_DB)
        self.file_loader = file_loader
        self.transform = transforms.Compose(
            [
                TruncatedInputfromMFB(),  # numpy array:(1, n_frames, n_dims)
                ToTensorInput(),  # torch tensor:(1, n_dims, n_frames)
            ]
        )

    def _load_data(self, data_DB):
        nb_speaker = len(set(data_DB["labels"]))

        return {
            key: np.array(data_DB.loc[data_DB["labels"] == key]["filename"])
            for key in range(nb_speaker)
        }

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if (self.max_iter is None) or (self.num_iter < self.max_iter):
            self.num_iter += 1
            images, labels = self.sample(self.nb_classes, self.nb_samples_per_class)

            return (self.num_iter - 1), (images, labels)
        else:
            raise StopIteration()

    def sample(self, nb_classes, nb_samples_per_class):

        picture_list = sorted(set(self.data.keys()))
        sampled_characters = random.sample(self.data.keys(), nb_classes)
        labels_and_images = []
        for (k, char) in enumerate(sampled_characters):
            label = picture_list[char]
            _imgs = self.data[char]
            _ind = random.sample(range(len(_imgs)), nb_samples_per_class)
            labels_and_images.extend(
                [(label, self.transform(self.file_loader(_imgs[i]))) for i in _ind]
            )
        arg_labels_and_images = []
        for i in range(self.nb_samples_per_class):
            for j in range(self.nb_classes):
                arg_labels_and_images.extend(
                    [labels_and_images[i + j * self.nb_samples_per_class]]
                )

        labels, images = zip(*arg_labels_and_images)
        images = torch.stack(images, dim=0)
        labels = torch.tensor(labels, dtype=torch.long)

        return images, labels