import tensorflow as tf
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

from data_io.get_data import view_samples

def train(net, dataset, epochs, batch_size, print_every=10, show_every=100, figsize=(3,3)):
    """Runs training for DCGAN network.

    Args:
        net: DCGAN network class
        dataset: tensor dataset
        epochs: number of training iterations
        batch_size: training instances per batch
        print_every: frequency of printing loss
        show_every: frequency of displaying generated images
        figsize: matplotlib figure size

    Returns:
        Array of losses and image samples during training.
    """
    saver = tf.train.Saver()
    steps = 0
    sample_z = np.random.uniform(-1, 1, size=(72, net.z_size))
    samples, losses = [], []
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for e in range(epochs):
            for batch_x in dataset.batches(batch_size):
                steps += 1
                
                batch_z = np.random.uniform(-1, 1, (batch_size, net.z_size))
                _ = sess.run(net.d_opt, feed_dict = {net.input_real: batch_x, net.input_z: batch_z})
                _ = sess.run(net.g_opt, feed_dict = {net.input_real: batch_x, net.input_z: batch_z})
                
                if steps%print_every == 0:
                    train_loss_d = sess.run(net.d_loss, feed_dict = {net.input_real: batch_x, net.input_z: batch_z})
                    train_loss_g = sess.run(net.g_loss, feed_dict = {net.input_z: batch_z, net.input_real: batch_x})
  
                    print("Epoch {}/{}...".format(e+1, epochs),
                          "Discriminator Loss: {:.4f}".format(train_loss_d),
                          "Generator Loss: {:.4f}".format(train_loss_g))
                    losses.append((train_loss_d, train_loss_g))
                
                if steps % show_every == 0:
                    gen_samples = sess.run(
                                    generator(net.input_z, 3, reuse=True, training=False),
                                    feed_dict = {net.input_z: sample_z})
                    samples.append(gen_samples)
                    _ = view_samples(-1, samples, 6, 12, figsize=figsize)
                    plt.show()
        saver.save(sess, './generator.ckpt')
        
    with open('samples.pkl', 'wb') as f:
        pkl.dump(samples, f)
        
    return losses, samples