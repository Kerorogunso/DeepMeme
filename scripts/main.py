import tensorflow as tf
import numpy as np
import os

from sklearn.model_selection import train_test_split
from datetime import datetime

from data_io.get_data import Dataset, get_memes_data, path_to_tensor
from train.train_dcgan import train
from models.dcgan import DCGAN

if __name__ == "__main__":
    real_size = (64,64,3)
    z_size = 100
    learning_rate = 0.0004
    batch_size = 128
    epochs = 10
    alpha = 0.2
    beta1 = 0.5
    dim = 64

    memes = get_memes_data('C:\\Users\\Albert\\Documents\\GitHub\\DeepMeme\\data\\me_irl')
    reshaped_memes = np.array([path_to_tensor(img_path, dim) for img_path in memes])

    now = datetime.utcnow().strftime('%Y%m%d%H%M%S')
    root_logdir = "tf_logs"
    logdir = "{}/run-{}/".format(root_logdir, now)

    net = DCGAN(real_size, z_size, learning_rate, alpha, beta1)
    generator_summary = tf.summary.scalar('g_loss', net.g_loss)
    discriminator_summary = tf.summary.scalar('d_loss', net.d_loss)
    file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
    print(tf.get_default_graph())

    reshaped_memes = reshaped_memes[:100] # start with 100 images only

    trainset, testset = train_test_split(reshaped_memes, test_size=0.2)
    dataset = Dataset(trainset, testset)
    losses, samples = train(net, dataset, epochs, batch_size, figsize=(10,5))