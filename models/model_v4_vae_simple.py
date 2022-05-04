import h5py
import tensorflow as tf
from tensorflow.python.keras.saving import hdf5_format
import numpy as np

from . import scalers, nn


@tf.function(experimental_relax_shapes=True)
def preprocess_features(features):
    # features:
    #   crossing_angle [-20, 20]
    #   dip_angle [-60, 60]
    #   drift_length [35, 290]
    #   pad_coordinate [40-something, 40-something]
    bin_fractions = features[:, -2:] % 1
    features = (features[:, :3] - tf.constant([[0.0, 0.0, 162.5]])) / tf.constant([[20.0, 60.0, 127.5]])
    return tf.concat([features, bin_fractions], axis=-1)


_f = preprocess_features

@tf.function
def img_loss(d_real, d_fake):
    return tf.reduce_mean(tf.reduce_sum(tf.keras.losses.mean_squared_error(d_real, d_fake), axis=(0, 1)))

@tf.function
def KL_div(mu, log_sigma):
    #https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians

    return tf.reduce_mean(-0.5 * tf.reduce_sum(1 + log_sigma - mu**2 - tf.exp(log_sigma), axis=1))

@tf.function
def get_val_metric_v(imgs, imgs_unscaled):
    """Returns a vector of gaussian fit results to the image.
    The components are: [mu0, mu1, sigma0^2, sigma1^2, covariance, integral]
    
    assert len(imgs.shape) == 3, 'get_val_metric_v: Wrong images dimentions'
    check1 = tf.get_static_value(tf.reduce_all(imgs >= 0))
    check2 = tf.get_static_value(tf.reduce_all(tf.reduce_any(imgs > 0, axis=(1, 2))))
    assert check1, 'get_val_metric_v: Negative image content'
    assert check2, 'get_val_metric_v: some images are empty'
    """

    #step 1
    imgs_n = tf.ones_like(imgs)
    imgs_n = tf.reshape(imgs_n, imgs_n.shape[1:]+imgs_n.shape[0])
    tf.multiply(imgs_n, tf.reduce_sum(imgs, [1, 2]))
    imgs_n = tf.reshape(imgs_n, imgs_n.shape[-1]+imgs_n.shape[:2])
    imgs_n = imgs / imgs_n

    #step 2
    indexes = np.fromfunction(
        lambda i, j: (i, j),
        shape=imgs.shape[1:],
    )
    prep2 = tf.expand_dims(imgs_n, axis=1)

    #step 3
    prep3 = tf.cast(tf.expand_dims(tf.stack([indexes[0], indexes[1]]), axis=0), tf.float32)

    #step 4
    prep4 = prep2 * prep3
    mu = tf.reduce_sum(prep4, [2, 3])

    #step 1
    prep1 = tf.stack([indexes[0]*indexes[0], indexes[1]*indexes[1], indexes[0]*indexes[1]])
    prep1 = tf.cast(tf.expand_dims(prep1, axis=0), tf.float32)

    #step 2
    prep2 = prep2 * prep1
    prep2 = tf.reduce_sum(prep2, [2, 3])

    #step 3
    prep3 = tf.transpose(tf.stack([mu[:, 0] ** 2, mu[:, 1] ** 2, mu[:, 0] * mu[:, 1]]))

    #step 4
    cov = prep2 - prep3

    integral = tf.expand_dims(tf.reduce_sum(imgs_unscaled, [1, 2]), axis=1)

    return mu, cov, integral


class Model_v4_VAE:
    def __init__(self, config):
        self.opt = tf.keras.optimizers.Adam(config['lr'])
        self.kl_lambda = config['kl_lambda']

        self.latent_dim = config['latent_dim']

        architecture_descr = config['architecture']
        self.encoder = nn.build_architecture(
            architecture_descr['encoder'], custom_objects_code=config.get('custom_objects', None)
        )
        self.decoder = nn.build_architecture(
            architecture_descr['decoder'], custom_objects_code=config.get('custom_objects', None)
        )

        self.step_counter = 0

        self.scaler = scalers.get_scaler(config['scaler'])
        self.pad_range = tuple(config['pad_range'])
        self.time_range = tuple(config['time_range'])
        self.data_version = config['data_version']
        self.enc_type = config['encoder_type']

        self.encoder.compile(optimizer=self.opt, loss='mean_squared_error', run_eagerly=True)
        self.decoder.compile(optimizer=self.opt, loss='mean_squared_error', run_eagerly=True)

    def load_encoder(self, checkpoint):
        self._load_weights(checkpoint, 'enc')

    def load_decoder(self, checkpoint):
        self._load_weights(checkpoint, 'dec')

    def _load_weights(self, checkpoint, enc_or_dec):
        if enc_or_dec == 'enc':
            network = self.encoder
            step_fn = self.training_step
        elif enc_or_dec == 'dec':
            network = self.decoder
            step_fn = self.training_step
        else:
            raise ValueError(enc_or_dec)

        model_file = h5py.File(checkpoint, 'r')
        if len(network.optimizer.weights) == 0 and 'optimizer_weights' in model_file:
            """
            # perform single optimization step to init optimizer weights
            features_shape = self.discriminator.inputs[0].shape.as_list()
            targets_shape = self.discriminator.inputs[1].shape.as_list()
            features_shape[0], targets_shape[0] = 1, 1
            step_fn(tf.zeros(features_shape), tf.zeros(targets_shape))
            """

        print(f'Loading {enc_or_dec} weights from {str(checkpoint)}')
        network.load_weights(str(checkpoint))
        """
        if 'optimizer_weights' in model_file:
            print('Also recovering the optimizer state')
            opt_weight_values = hdf5_format.load_optimizer_weights_from_hdf5_group(model_file)
            network.optimizer.set_weights(opt_weight_values)
        """

    @tf.function
    def sample(self, mu, log_sigma):
        epsilons = tf.random.normal(mu.shape)
        return mu + epsilons * tf.exp(log_sigma*.5)

    @tf.function
    def make_fake(self, features):
        size = tf.shape(features)[0]
        latent_input = tf.random.normal(shape=(size, self.latent_dim), dtype='float32')
        return self.decoder(latent_input)
    
    @tf.function
    def encode(self, x):
        if self.enc_type == 'fc':
            x_shape = tf.shape(x)
            latent_input = tf.reshape(x, shape=(x_shape[0], x_shape[1]*x_shape[2]))
            res = latent_input
        elif self.enc_type == 'conv':
            res = x
        return self.encoder(res)

    @tf.function
    def decode(self, z):
        return self.decoder(z)

    @tf.function
    def calculate_losses(self, feature_batch, target_batch):
        encoded_batch = self.encode(target_batch)
        mu, log_sigma = encoded_batch[:,0,:], encoded_batch[:,1,:]

        KL = KL_div(mu, log_sigma)
        z = self.sample(mu, log_sigma)
        res = self.decode(z)

        loss_img = img_loss(target_batch, res)
        loss_kl = KL * self.kl_lambda

        return {'loss': loss_img+loss_kl}

    @tf.function
    def training_step(self, feature_batch, target_batch):
        self.step_counter += 1
        
        feature_batch = tf.convert_to_tensor(feature_batch)
        target_batch = tf.convert_to_tensor(target_batch)
        vars = self.decoder.trainable_variables
        vars += self.encoder.trainable_variables

        with tf.GradientTape() as t:
            losses = self.calculate_losses(feature_batch, target_batch)

        grads = t.gradient(losses['loss'], vars)
        self.opt.apply_gradients(zip(grads, vars))
        return losses
