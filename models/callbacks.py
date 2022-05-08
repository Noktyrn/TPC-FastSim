import tensorflow as tf

from metrics import make_images_for_model


class SaveModelCallback:
    def __init__(self, model, path, save_period):
        self.model = model
        self.path = path
        self.save_period = save_period

    def __call__(self, step):
        if step % self.save_period == 0:
            print(f'Saving model on step {step} to {self.path}')
            self.model.generator.save(str(self.path.joinpath("generator_{:05d}.h5".format(step))))
            self.model.discriminator.save(str(self.path.joinpath("discriminator_{:05d}.h5".format(step))))


class WriteHistSummaryCallback:
    def __init__(self, model, sample, save_period, writer):
        self.model = model
        self.sample = sample
        self.save_period = save_period
        self.writer = writer

    def __call__(self, step):
        if step % self.save_period == 0:
            images, images1, img_amplitude, chi2 = make_images_for_model(self.model, sample=self.sample, calc_chi2=True)
            with self.writer.as_default():
                tf.summary.scalar("chi2", chi2, step)

                for k, img in images.items():
                    tf.summary.image(k, img, step)
                for k, img in images1.items():
                    tf.summary.image("{} (amp > 1)".format(k), img, step)
                tf.summary.image("log10(amplitude + 1)", img_amplitude, step)


class ScheduleLRCallback:
    def __init__(self, model, func_gen, func_disc, writer):
        self.model = model
        self.func_gen = func_gen
        self.func_disc = func_disc
        self.writer = writer

    def __call__(self, step):
        self.model.disc_opt.lr.assign(self.func_disc(step))
        self.model.gen_opt.lr.assign(self.func_gen(step))
        with self.writer.as_default():
            tf.summary.scalar("discriminator learning rate", self.model.disc_opt.lr, step)
            tf.summary.scalar("generator learning rate", self.model.gen_opt.lr, step)

class VAE_ScheduleLRCallback:
    def __init__(self, model, func, writer):
        self.model = model
        self.func = func
        self.writer = writer

    def __call__(self, step):
        self.model.opt.lr.assign(self.func(step))
        with self.writer.as_default():
            tf.summary.scalar("learning rate", self.model.opt.lr, step)

class VAE_SaveModelCallback:
    def __init__(self, model, path, save_period):
        self.model = model
        self.path = path
        self.save_period = save_period

    def __call__(self, step):
        if step % self.save_period == 0:
            print(f'Saving model on step {step} to {self.path}')
            self.model.encoder.save(str(self.path.joinpath("encoder_{:05d}.h5".format(step))))
            self.model.decoder.save(str(self.path.joinpath("decoder_{:05d}.h5".format(step))))

class VAE_PlateauScheduleLRCallback:
    def __init__(self, model, func, writer):
        self.model = model
        self.func = func
        self.writer = writer

    def __call__(self, step, loss):
        """
        self.model.opt.lr.assign(self.func(step))
        with self.writer.as_default():
            tf.summary.scalar("learning rate", self.model.opt.lr, step)
        """
        self.model.opt.lr.assign(self.func(loss))
        with self.writer.as_default():
            tf.summary.scalar("learning rate", self.model.opt.lr, step)

def get_scheduler(lr, lr_decay):
    if isinstance(lr_decay, str):
        return eval(lr_decay)

    def schedule_lr(step):
        return lr / ((step+1)**lr_decay)

    return schedule_lr

class LR_Reduce_On_Plateau:
    def __init__(self, lr, factor=1e-1, patience=5, delta=1e-4):
        self.counter = 0
        self.patience = patience
        self.delta = delta
        self.factor = factor
        self.previous = 0
        self.lr = lr
    
    def __call__(self, loss):
        diff = self.previous - loss
        if diff < self.delta:
            self.counter += 1
        else:
            self.counter = 0
        
        if self.counter == self.patience:
            self.lr *= self.factor
            self.counter = 0
        self.previous = loss
        
        return self.lr
        
