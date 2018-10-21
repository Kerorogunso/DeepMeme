import numpy as np

from sklearn.datasets import load_files
from keras.preprocessing import image

def get_memes_data(path):
    """Load the memes dataset as array when given a file path.

    Args:
        path: the path to the dataset
    Returns:
        Numpy array of the loaded dataset.

    """
    data = load_files(path)
    memes = np.array(data['filenames'])
    return memes

def path_to_tensor(img_path, dim):
    """Load image path and convert it to ndarray.

    Args:
        img_path: path to image
        dim: target image width
    Returns:
        Tensor of the image
    """

    # Load RGB image
    img = image.load_img(img_path, target_size=(dim,dim))
    # Convert to (dim, dim,3) tensor
    x = image.img_to_array(img)
    return x

def scale(x, feature_range=(-1,1)):
    """Normalizes pixel intensities.

    Args:
        x: tensor for scaling
        feature_range: the resulting range for pixels
    Returns:
        Tensor with entries rescaled.
    """
    x = ((x - x.min())/(255 - x.min()))
    
    minimum, maximum = feature_range
    x = x * (maximum - minimum) + minimum
    return x

class Dataset:
    def __init__(self, train, test, val_frac=0.5, shuffle = False, scale_func=None):
        split_idx = int(len(test) * (1 - val_frac))
        self.test_x, self.valid_x = test[:,:,:,:split_idx], test[:,:,:,split_idx:]

        self.train_x = train
        
        if scale_func is None:
            self.scaler = scale
        else:
            self.scaler = scale_func
        self.shuffle = shuffle
        
    def batches(self, batch_size):
        """Batches dataset for training.

        Args:
            batch_size: number of training instances per batch
        Returns:
            Dataset batches in the form of a generator
        """    
        if self.shuffle:
            idx = np.arange(len(dataset.train_x))
            np.random.shuffle(idx)
            self.train_x = self.train_x[idx]
            
        n_batches = len(self.train_x) // batch_size
        for ii in range(0, len(self.train_x), batch_size):
            x = self.train_x[ii:ii+batch_size]
            
            yield self.scaler(x)

def view_samples(epoch, samples, nrows, ncols, figsize=(3,3)):
    """View training samples at various stages of training.

    Args:
        epoch: epoch number during training
        samples: number of images to display
        nrows: number of images per row
        ncols: number of images per column
        figsize: matplotlib figure size
    
    Returns:
        The figure and axes for the plots.
    """
    fig, axes = plt.subplots(figsize=figsize, nrows = nrows, ncols = ncols,
                            sharey=True, sharex=True)
    for ax, img in zip(axes.flatten(), samples[epoch]):
        ax.axis('off')
        img = ((img - img.min())*255 / (img.max() - img.min())).astype(np.uint8)
        ax.set_adjustable('box-forced')
        im = ax.imshow(img, aspect='equal')
        
    plt.subplots_adjust(wspace=0, hspace=0)
    return fig, axes