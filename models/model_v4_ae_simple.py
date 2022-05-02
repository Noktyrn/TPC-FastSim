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


def ae_loss(d_real, d_fake):
    return tf.reduce_mean(tf.reduce_sum(tf.keras.losses.mean_squared_error(d_real, d_fake), axis=(0, 1)))


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
        z = self.encode(target_batch)
        res_f = self.decode(z)
        return {'loss': ae_loss(target_batch, res_f)}

    @tf.function
    def training_step(self, feature_batch, target_batch):
        target_batch = tf.convert_to_tensor(target_batch)

        with tf.GradientTape() as t:
            losses = self.calculate_losses(feature_batch, target_batch)

        train_vars = self.decoder.trainable_variables + self.encoder.trainable_variables

        grads = t.gradient(losses['loss'], train_vars)
        self.opt.apply_gradients(zip(grads, train_vars))
        self.step_counter += 1
        return losses
