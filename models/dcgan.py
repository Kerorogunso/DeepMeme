import tensorflow as tf

dim = 64

def model_inputs(real_dim, z_dim):
    """Create tensorflow variables for model input.

    Args:
        real_dim: tensor shape for the real input
        z_dim: tensor shape for the random variable

    Returns:
        Placeholder variables for the real input and random variable for the generator.
    """
    inputs_real= tf.placeholder(tf.float32, (None, *real_dim), name = 'input_real')
    inputs_z = tf.placeholder(tf.float32, (None, z_dim), name='input_z')
    
    return inputs_real, inputs_z

def generator(z, output_dim, g_hidden=256, reuse=False, alpha=0.2, training=True):
    """Creates the generator network of DCGAN.
    
    Args:
        z: random variable the image is generated from
        output_dim: output dimensions of the generator
        g_hidden: number of hidden units in the first layer
        reuse: flag for reusing variable names
        alpha: coefficient for leaky relu
        training: flag controlling batch normalization when training

    Returns:
        The output layer of the generator.
    """

    with tf.variable_scope('generator', reuse=reuse):
        # First fully connected layer
        x1 = tf.layers.dense(z, (dim//4)*(dim//4)*g_hidden)
        
        # Reshape to start convolutional stack
        x1 = tf.reshape(x1, (-1, (dim//4), (dim//4), g_hidden))
        x1 = tf.layers.batch_normalization(x1, training=training)
        x1 = tf.maximum(alpha * x1, x1)
        # (dim/4)x(dim/4)x(g_hidden) now
        
        x2 = tf.layers.conv2d_transpose(x1, g_hidden//2, 5, strides=2, padding='same')
        x2 = tf.layers.batch_normalization(x2, training=training)
        x2 = tf.maximum(alpha * x2, x2)
        # (dim//2)x(dim//2)x(g_hidden//2) now
        
        x3 = tf.layers.conv2d_transpose(x2, g_hidden//4, 5, strides=2, padding='same')
        x3 = tf.layers.batch_normalization(x3, training=training)
        x3 = tf.maximum(alpha * x3, x3)
        # (dim)x(dim)x(g_hidden//4) now
        
        # Output layer
        logits = tf.layers.conv2d_transpose(x3, output_dim, 5, strides=2, padding = 'same')
        # (dim)x(dim)x3
        out = tf.tanh(logits)
        
        return out

def discriminator(x, d_hidden = 64, reuse = False, alpha = 0.2, training=True):
    """Creates the discriminator network.

    Args:
        x: input to the discriminator
        d_hidden: number of hidden units in the first layer
        reuse: flag for reusing variable names
        alpha: coefficient for leaky relu
        training: flag for batch normalization

    Returns:
        Output layer and logits for the discriminator network.
    """
    with tf.variable_scope('discriminator', reuse= reuse):
        # Input layer is (dim)x(dim)x3
        x1 = tf.layers.conv2d(x, d_hidden, 5, strides=2, padding='same')
        relu1 = tf.maximum(alpha * x1, x1)
        # (dim//2)x(dim//2)x(d_hidden)
        
        x2 = tf.layers.conv2d(relu1, d_hidden*2, 5, strides=2, padding='same')
        bn2 = tf.layers.batch_normalization(x2, training=training)
        relu2 = tf.maximum(alpha * bn2, bn2)
        # (dim//4)x(dim//4)(d_hidden*2)
        
        x3 = tf.layers.conv2d(relu2, d_hidden*4, 5, strides=2, padding='same')
        bn3 = tf.layers.batch_normalization(x3, training=training)
        relu3 = tf.maximum(alpha * bn3, bn3)
        # (dim//8)x(dim//8)x(d_hidden*4)
        
        flat = tf.reshape(relu3, (-1, (dim//8)*(dim//8)*(d_hidden*4)))
        logits = tf.layers.dense(flat, 1)
        out = tf.sigmoid(logits)
        
        return out, logits

def model_loss(input_real, input_z, output_dim, alpha=0.2):
    """Creates model loss functions for the generator and discriminator networks.

    Args:
        input_real: real input to the discriminator
        input_z: random variable input to the generator
        output_dim: output dimension of the generator
        alpha: coefficient for leaky relu

    Returns:
        Discriminator loss and generator loss
    """
    g_out = generator(z=input_z, output_dim=output_dim, alpha=alpha, training=True)
    d_model_real, d_logits_real = discriminator(x=input_real, alpha=alpha, training=True)
    d_model_fake, d_logits_fake = discriminator(x=g_out, alpha=alpha, reuse=True, training=True)
    
    g_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_logits_fake, labels=tf.ones_like(d_model_fake)))
    d_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits = d_logits_real, labels = tf.ones_like(d_model_real)))
    d_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits = d_logits_fake, labels = tf.zeros_like(d_model_fake)))
    
    d_loss = d_loss_real + d_loss_fake
    
    return d_loss, g_loss

def model_opt(d_loss, g_loss, learning_rate, beta1):
    """Creates model optimizers for the generator and discriminator networks.

    Args:
        d_loss: loss for the discriminator
        g_loss: loss for the generator

    Returns:
        Discriminator and generator optimizers
    """
    t_vars = tf.trainable_variables()
    g_vars = [var for var in t_vars if "generator" in var.name]
    d_vars = [var for var in t_vars if "discriminator" in var.name]
    
    d_optimizer = tf.train.AdamOptimizer(learning_rate, beta1)
    g_optimizer = tf.train.AdamOptimizer(learning_rate, beta1)
    
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        d_operator = d_optimizer.minimize(d_loss, var_list= d_vars)
        g_operator = g_optimizer.minimize(g_loss, var_list= g_vars)
    
    return d_operator, g_operator

class DCGAN:
    def __init__(self, real_size, z_size, learning_rate, alpha=0.2, beta1=0.5):
        self.real_size = real_size
        self.z_size = z_size
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.beta1 = beta1
        tf.reset_default_graph()
        self.input_real, self.input_z = model_inputs(real_size, z_size)
        self.d_loss, self.g_loss = model_loss(self.input_real, self.input_z, real_size[2], alpha=alpha)
        self.d_opt, self.g_opt = model_opt(self.d_loss, self.g_loss, learning_rate, beta1)