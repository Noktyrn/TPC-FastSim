import h5py
import tensorflow as tf
from tensorflow.python.keras.saving import hdf5_format


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
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(d_real-d_fake), axis=(1, 2)))
    return loss

@tf.function
def conv_loss(z_real, z_conv):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(z_real-z_conv), axis=(1)))
    return loss


def KL_div(mu, log_sigma):
    #https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians

    return tf.reduce_mean(-0.5 * tf.reduce_sum(1 + log_sigma - mu**2 - tf.exp(log_sigma), axis=1))


class Model_v4_AE:
    def __init__(self, config):
        self.opt = tf.keras.optimizers.Adam(config['lr'])

        architecture_descr = config['architecture']
        self.encoder = nn.build_architecture(
            architecture_descr['encoder'], custom_objects_code=config.get('custom_objects', None), 
            input_shape = config.get('input_shape', None)
        )
        self.decoder = nn.build_architecture(
            architecture_descr['decoder'], custom_objects_code=config.get('custom_objects', None)
        )

        self.step_counter = 0
        self.latent_dim = config['latent_dim']

        self.scaler = scalers.get_scaler(config['scaler'])
        self.pad_range = tuple(config['pad_range'])
        self.time_range = tuple(config['time_range'])
        self.data_version = config['data_version']
        self.enc_type = config['encoder_type']

        self.encoder.compile(optimizer=self.opt, loss='mean_squared_error')
        self.decoder.compile(optimizer=self.opt, loss='mean_squared_error')

    def save_weights(self, path, step):
        self.encoder.save(str(path.joinpath("encoder_{:05d}.h5".format(step))))
        self.decoder.save(str(path.joinpath("decoder_{:05d}.h5".format(step))))
    
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
        return mu + epsilons * tf.exp(log_sigma / 2)

    @tf.function
    def make_fake(self, features):
        size = tf.shape(features)[0]
        latent_input = tf.random.normal(shape=(size, self.latent_dim), dtype='float32')
        return self.decoder(tf.concat([_f(features), latent_input], axis=-1))
    
    @tf.function
    def encode(self, features, x):
        if self.enc_type == 'fc':
            size = tf.shape(features)
            x_shape = tf.shape(x)
            latent_input = tf.reshape(x, shape=(size[0], x_shape[1]*x_shape[2]))
            res = tf.concat([_f(features), latent_input], axis=-1)
        elif self.enc_type == 'conv':
            res = [_f(features), x]
        return self.encoder(res)

    @tf.function
    def decode(self, features, z):
        z_with_features = tf.concat([_f(features), z], axis=-1)
        return self.decoder(z_with_features)

    @tf.function
    def calculate_losses(self, feature_batch, target_batch):
        z = self.encode(feature_batch, target_batch)
        res_f = self.decode(feature_batch, z)
        #ae_l = ae_loss(_f(feature_batch), encoded_batch)
        #tf.print(ae_l)
        loss = img_loss(target_batch, res_f)
        return {'loss': loss}

    @tf.function
    def training_step(self, feature_batch, target_batch):
        feature_batch = tf.convert_to_tensor(feature_batch)
        target_batch = tf.convert_to_tensor(target_batch)

        with tf.GradientTape() as t:
            losses = self.calculate_losses(feature_batch, target_batch)

        train_vars = self.decoder.trainable_variables + self.encoder.trainable_variables

        grads = t.gradient(losses['loss'], train_vars)
        self.opt.apply_gradients(zip(grads, train_vars))
        self.step_counter += 1
        return losses
