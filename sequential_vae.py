from abstract_network import *
from scipy import misc


class SequentialVAE(Network):
    def __init__(self, dataset, batch_size, name=None):
        Network.__init__(self, dataset)
        if name is None or name == "":
            self.name = "sequential_vae_%s" % dataset.name
        else:
            self.name = name
        self.dataset = dataset
        self.batch_size = batch_size
        self.data_dims = dataset.data_dims

        self.fs = [self.data_dims[0], self.data_dims[0] // 2, self.data_dims[0] // 4, self.data_dims[0] // 8,
                   self.data_dims[0] // 16]
        # Increasing this is generally beneficial if you have more memory
        self.cs = [self.data_dims[-1], 48, 96, 192, 384, 512]

        # Share weights between generator steps. Set this to true for homogeneous Markov chain.
        self.share = False
        # Because the latent samples q(z_1, ..., z_T) are not independent. This explicitly models that dependence.
        # In general this gives a marginal improvement on sample quality
        self.use_latent_pred = False
        # Some external information to condition on.
        # For example this can do conditional generation given label information
        # To activate, replace this with a placeholder.
        # Supply the value for that placeholder during both training and sampling
        self.condition = None

        # Configuration for different netnames. Add your own if needed
        if self.name == "sequential_vae_celebA":
            self.ladder_dims = [10, 10, 10, 10]
            self.latent_dim = np.sum(self.ladder_dims)
            self.steps = 8
            self.generator = self.generator_ladder
            self.inference = self.inference_ladder
        elif self.name == "sequential_vae_celebA_ladder_pred":
            self.ladder_dims = [10, 10, 10, 10]
            self.latent_dim = np.sum(self.ladder_dims)
            self.steps = 8
            self.use_latent_pred = True
            self.generator = self.generator_ladder
            self.inference = self.inference_ladder
        elif self.name == "sequential_vae_lsun":
            self.ladder_dims = [20, 30, 30, 30]
            self.latent_dim = np.sum(self.ladder_dims)
            self.steps = 8
            self.generator = self.generator_ladder
            self.inference = self.inference_ladder
        elif self.name == "sequential_vae_lsun_ladder_pred":
            self.cs = [self.data_dims[-1], 64, 128, 256, 512, 1024]
            self.ladder_dims = [20, 20, 20, 20]
            self.latent_dim = np.sum(self.ladder_dims)
            self.steps = 8
            self.use_latent_pred = True
            self.generator = self.generator_ladder
            self.inference = self.inference_ladder
        else:
            print("Unknown network name %s" % self.name)
            exit(-1)

        self.input_placeholder = tf.placeholder(shape=[None]+self.data_dims, dtype=tf.float32, name="input_placeholder")
        self.target_placeholder = tf.placeholder(shape=[None]+self.data_dims, dtype=tf.float32, name="target_placeholder")

        self.reg_coeff = tf.placeholder_with_default(1.0, shape=[], name="regularization_coeff")
        self.latents = []
        self.tsamples = []
        self.gsamples = []
        self.regularization = 0.0
        self.loss = 0.0
        self.final_loss = None
        tsample = None
        gsample = None
        latent_mean_pred, latent_stddev_pred = None, None
        glatent_mean_pred, glatent_stddev_pred = None, None
        for step in range(self.steps):
            latent_placeholder = tf.placeholder(shape=[None, self.latent_dim], dtype=tf.float32, name="latent_ph%d" % step)
            self.latents.append(latent_placeholder)
            if step == 0:
                gsample = tf.random_uniform(shape=tf.pack([tf.shape(self.input_placeholder)[0]] + self.data_dims))
                self.gsamples.append(gsample)

            latent_mean, latent_stddev = self.inference(self.input_placeholder, step)
            latent_sample = latent_mean + tf.mul(latent_stddev,
                                                 tf.random_normal(tf.pack([tf.shape(self.input_placeholder)[0], self.latent_dim])))

            # Predict the latent state of the next step.
            # This can be used to explicitly capture dependence between latent code
            if latent_mean_pred is not None:
                pred_loss = tf.log(latent_stddev_pred) - tf.log(latent_stddev) + \
                            tf.div(tf.square(latent_stddev), 2 * tf.square(latent_stddev_pred)) + \
                            tf.div(tf.square(latent_mean - latent_mean_pred), 2 * tf.square(latent_stddev_pred))
                pred_loss = tf.reduce_sum(pred_loss) / self.batch_size - 0.5 * self.latent_dim
                tf.summary.scalar("latent_pred_loss%d" % step, pred_loss)
                self.loss += pred_loss

            # If we make a prediction then transform input white Gaussian to the Gaussian we predicted
            if glatent_mean_pred is not None:
                external_latent = glatent_mean_pred + tf.mul(glatent_stddev_pred, latent_placeholder)
            else:
                external_latent = latent_placeholder

            # Obtain prediction for next step
            if self.use_latent_pred:
                if self.share:
                    latent_mean_pred, latent_stddev_pred = \
                        self.latent_code_generator(latent_sample, step=None, reuse=(step != 0), condition=self.condition)
                    glatent_mean_pred, glatent_stddev_pred = \
                        self.latent_code_generator(external_latent, step=None, reuse=True, condition=self.condition)
                else:
                    latent_mean_pred, latent_stddev_pred = \
                        self.latent_code_generator(latent_sample, step=step, condition=self.condition)
                    glatent_mean_pred, glatent_stddev_pred = \
                        self.latent_code_generator(external_latent, step=step, reuse=True, condition=self.condition)
            # Generate samples
            if step == 0:
                tsample = self.generator(None, latent_sample, step, condition=self.condition)
                gsample = self.generator(None, external_latent, step, reuse=True, condition=self.condition)
            else:
                if self.share:
                    tsample, ratio = self.generator(tsample, latent_sample, None, reuse=(step != 1), condition=self.condition)
                    gsample, _ = self.generator(gsample, external_latent, None, reuse=True, condition=self.condition)
                else:
                    tsample, ratio = self.generator(tsample, latent_sample, step, condition=self.condition)
                    gsample, _ = self.generator(gsample, external_latent, step, reuse=True, condition=self.condition)
                tf.summary.scalar("resnet_gate_weight%d" % step, tf.reduce_mean(ratio))
            self.tsamples.append(tsample)
            self.gsamples.append(gsample)

            const1 = math.log(math.sqrt(2 * math.pi))
            const2 = math.log(math.sqrt(2 * math.pi * math.e))
            regularization_loss = tf.reduce_sum(-tf.log(latent_stddev) +
                                                0.5 * tf.square(latent_stddev) +
                                                0.5 * tf.square(latent_mean)) / self.batch_size - const2 + const1
            step_loss = tf.reduce_sum(tf.square(tsample - self.target_placeholder)) / self.batch_size
            self.loss += 16 * step_loss + regularization_loss * self.reg_coeff
            self.final_loss = step_loss

            tf.summary.scalar("reconstruction_loss%d" % step, step_loss)
            tf.summary.scalar("regularization_loss%d" % step, regularization_loss)

            tsample = tf.stop_gradient(tsample)
        tf.summary.scalar("loss", self.loss)

        self.merged_summary = tf.summary.merge_all()
        self.iteration = 0
        # gen_vars = [var for var in tf.global_variables() if 'generative' in var.name]
        self.train_op = tf.train.AdamOptimizer(0.0002).minimize(self.loss)

        self.init_network(restart=False)
        self.print_network()
        self.read_only = False

        self.mc_fig, self.mc_ax = None, None

    def train(self, batch_input, batch_target, condition=None):
        self.iteration += 1
        feed_dict = {self.input_placeholder: batch_input,
                     self.reg_coeff: 1 - math.exp(-self.iteration / 10000.0),
                     self.target_placeholder: batch_target}
        if self.condition is not None:
            feed_dict[self.condition] = condition
        train_return = self.sess.run([self.train_op, self.loss, self.final_loss], feed_dict=feed_dict)
        if self.iteration % 2000 == 0:
            self.save_network()
        if self.iteration % 100 == 0:
            summary = self.sess.run(self.merged_summary, feed_dict=feed_dict)
            self.writer.add_summary(summary, self.iteration)
        return train_return[2] / self.data_dims[0] / self.data_dims[1]

    def test(self, batch_input, condition=None):
        feed_dict = {self.input_placeholder: batch_input}
        if self.condition is not None:
            feed_dict[self.condition] = condition
        train_return = self.sess.run(self.tsamples[-1],
                                     feed_dict=feed_dict)
        return train_return

    def random_latent_code(self):
        pass

    def generate_mc_samples(self, batch_input, batch_size=None, condition=None):
        if batch_size is None:
            batch_size = self.batch_size
        feed_dict = dict()
        if self.condition is not None:
            feed_dict[self.condition] = condition

        for i in range(self.steps):
            feed_dict[self.latents[i]] = np.random.normal(size=(batch_size, self.latent_dim))
        # This is needed because for some versions batch_norm requires input from shared weight networks
        feed_dict[self.input_placeholder] = batch_input
        output = self.sess.run(self.gsamples, feed_dict=feed_dict)
        return output

    def conditioned_mc_samples(self, batch_input, condition=None):
        feed_dict = dict()
        if self.condition is not None:
            feed_dict[self.condition] = condition
        feed_dict[self.input_placeholder] = batch_input
        output = self.sess.run(self.tsamples, feed_dict=feed_dict)
        return output

    def visualize(self, epoch, batch_size=10, condition=None, use_gui=True):
        if use_gui is True and self.mc_fig is None:
            self.mc_fig, self.mc_ax = plt.subplots(1, 2)

        for i in range(2):
            if i == 0:
                bx = self.dataset.next_batch(batch_size)
                z = self.generate_mc_samples(bx, batch_size, condition=condition)
            else:
                bx = self.dataset.next_batch(batch_size)
                z = self.conditioned_mc_samples(bx, condition=condition)
                z = [bx] + z
            v = np.zeros([z[0].shape[0] * self.data_dims[0], len(z) * self.data_dims[1], self.data_dims[2]])
            for b in range(0, z[0].shape[0]):
                for t in range(0, len(z)):
                    v[b*self.data_dims[0]:(b+1)*self.data_dims[0], t*self.data_dims[1]:(t+1)*self.data_dims[1]] = self.dataset.display(z[t][b])

            if use_gui is True:
                self.mc_ax[i].cla()
                if self.data_dims[-1] == 1:
                    self.mc_ax[i].imshow(v[:, :, 0], cmap='gray')
                else:
                    self.mc_ax[i].imshow(v)
                self.mc_ax[i].xaxis.set_visible(False)
                self.mc_ax[i].yaxis.set_visible(False)
                if i == 0:
                    self.mc_ax[i].set_title("test")
                else:
                    self.mc_ax[i].set_title("train")

            folder_name = 'models/%s/samples' % self.name
            if not os.path.isdir(folder_name):
                os.makedirs(folder_name)

            if v.shape[-1] == 1:
                v = v[:, :, 0]

            if i == 0:
                misc.imsave(os.path.join(folder_name, 'test_epoch%d.png' % epoch), v)
                misc.imsave(os.path.join(folder_name, 'test_current.png'), v)
            else:
                misc.imsave(os.path.join(folder_name, 'train_epoch%d.png' % epoch), v)
                misc.imsave(os.path.join(folder_name, 'train_current.png'), v)
        if use_gui is True:
            plt.draw()
            plt.pause(0.01)

    def latent_code_generator(self, latent, condition=None, step=None, reuse=False):
        if step is None:
            scope_name = "latent_gen"
        else:
            scope_name = "latent_gen%d" % step
        with tf.variable_scope(scope_name) as scope:
            if reuse:
                scope.reuse_variables()
            if condition is not None:
                latent = tf.concat(1, [latent, condition])
            fc1 = fc_bn_lrelu(latent, 1024)
            fc2 = fc_bn_lrelu(fc1, 1024)
            fc3 = fc_bn_lrelu(fc2, 1024)
            out_mean = tf.contrib.layers.fully_connected(fc3, self.latent_dim, activation_fn=tf.identity)
            out_stddev = tf.contrib.layers.fully_connected(fc3, self.latent_dim, activation_fn=tf.sigmoid)
            return out_mean, out_stddev

    def inference_ladder(self, inputs, step, reuse=False):
        with tf.variable_scope("inference%d" % step) as scope:
            if reuse:
                scope.reuse_variables()
            iconv1 = conv2d_bn_lrelu(inputs, self.cs[1], [4, 4], 2)     # 32x32
            iconv2 = conv2d_bn_lrelu(iconv1, self.cs[1], [4, 4], 1)

            ladder0 = tf.reshape(iconv2, [-1, np.prod(iconv2.get_shape().as_list()[1:])])
            ladder0_mean = tf.contrib.layers.fully_connected(ladder0, self.ladder_dims[0], activation_fn=tf.identity)
            ladder0_stddev = tf.contrib.layers.fully_connected(ladder0, self.ladder_dims[0], activation_fn=tf.sigmoid)

            iconv3 = conv2d_bn_lrelu(iconv2, self.cs[2], [4, 4], 2)    # 16x16
            iconv4 = conv2d_bn_lrelu(iconv3, self.cs[2], [4, 4], 1)

            ladder1 = tf.reshape(iconv4, [-1, np.prod(iconv4.get_shape().as_list()[1:])])
            ladder1_mean = tf.contrib.layers.fully_connected(ladder1, self.ladder_dims[1], activation_fn=tf.identity)
            ladder1_stddev = tf.contrib.layers.fully_connected(ladder1, self.ladder_dims[1], activation_fn=tf.sigmoid)

            iconv5 = conv2d_bn_lrelu(iconv4, self.cs[3], [4, 4], 2)    # 8x8
            iconv6 = conv2d_bn_lrelu(iconv5, self.cs[3], [4, 4], 1)

            ladder2 = tf.reshape(iconv6, [-1, np.prod(iconv6.get_shape().as_list()[1:])])
            ladder2_mean = tf.contrib.layers.fully_connected(ladder2, self.ladder_dims[2], activation_fn=tf.identity)
            ladder2_stddev = tf.contrib.layers.fully_connected(ladder2, self.ladder_dims[2], activation_fn=tf.sigmoid)

            iconv7 = conv2d_bn_lrelu(iconv6, self.cs[4], [4, 4], 2)    # 4x4
            iconv7 = tf.reshape(iconv7, [-1, np.prod(iconv7.get_shape().as_list()[1:])])
            ifc1 = fc_bn_lrelu(iconv7, self.cs[5])
            ladder3_mean = tf.contrib.layers.fully_connected(ifc1, self.ladder_dims[3], activation_fn=tf.identity)
            ladder3_stddev = tf.contrib.layers.fully_connected(ifc1, self.ladder_dims[3], activation_fn=tf.sigmoid)

            latent_mean = tf.concat(1, [ladder0_mean, ladder1_mean, ladder2_mean, ladder3_mean])
            latent_stddev = tf.concat(1, [ladder0_stddev, ladder1_stddev, ladder2_stddev, ladder3_stddev])
            return latent_mean, latent_stddev

    def combine_noise(self, latent, ladder, name="default"):
        method = 'concat'
        if method is 'concat':
            return tf.concat(len(latent.get_shape())-1, [latent, ladder])
        elif method is 'add':
            return latent + ladder
        elif method is 'gated_add':
            gate = tf.get_variable("gate", shape=ladder.get_shape()[1:], initializer=tf.constant_initializer(0.1))
            tf.histogram_summary(name + "_noise_gate", gate)
            return latent + tf.mul(gate, ladder)

    def generator_ladder(self, inputs, latent, step=None, reuse=False, condition=None):
        if step is None:
            scope_name = "generative"
        else:
            scope_name = "generative%d" % step
        with tf.variable_scope(scope_name) as gs:
            if reuse:
                gs.reuse_variables()

            ladder0, ladder1, ladder2, ladder3 = tf.split_v(latent, self.ladder_dims, 1)
            # Manually set the size of splitted tensors because in some versions of tensowflow this is not automatic
            ladder0 = tf.reshape(ladder0, [-1, self.ladder_dims[0]])
            ladder1 = tf.reshape(ladder1, [-1, self.ladder_dims[1]])
            ladder2 = tf.reshape(ladder2, [-1, self.ladder_dims[2]])
            ladder3 = tf.reshape(ladder3, [-1, self.ladder_dims[3]])
            if condition is not None:
                ladder3 = tf.concat(1, [ladder3, condition])
                print("Add labels")
            ladder0 = fc_bn_lrelu(ladder0, int(self.fs[1] * self.fs[1] * self.cs[1]))
            ladder0 = tf.reshape(ladder0, [-1, self.fs[1], self.fs[1], self.cs[1]])
            ladder1 = fc_bn_lrelu(ladder1, int(self.fs[2] * self.fs[2] * self.cs[2]))
            ladder1 = tf.reshape(ladder1, [-1, self.fs[2], self.fs[2], self.cs[2]])
            ladder2 = fc_bn_lrelu(ladder2, int(self.fs[3] * self.fs[3] * self.cs[3]))
            ladder2 = tf.reshape(ladder2, [-1, self.fs[3], self.fs[3], self.cs[3]])
            ladder3 = fc_bn_lrelu(ladder3, self.cs[5])
            ladder3 = tf.reshape(ladder3, [-1, self.cs[5]])

            if inputs is not None:
                iconv1 = conv2d_bn_lrelu(inputs, self.cs[1], [4, 4], 2)    # 32x32
                iconv2 = conv2d_bn_lrelu(iconv1, self.cs[1], [4, 4], 1)
                iconv3 = conv2d_bn_lrelu(iconv2, self.cs[2], [4, 4], 2)    # 16x16
                iconv4 = conv2d_bn_lrelu(iconv3, self.cs[2], [4, 4], 1)
                iconv5 = conv2d_bn_lrelu(iconv4, self.cs[3], [4, 4], 2)    # 8x8
                iconv6 = conv2d_bn_lrelu(iconv5, self.cs[3], [4, 4], 1)
                iconv7 = conv2d_bn_lrelu(iconv6, self.cs[4], [4, 4], 2)    # 4x4
                iconv7 = tf.reshape(iconv7, [-1, np.prod(iconv7.get_shape().as_list()[1:])])
                ifc1 = fc_bn_lrelu(iconv7, self.cs[5])
                ifc2 = fc_bn_lrelu(ifc1, self.cs[5])
                ladder3 = self.combine_noise(ifc2, ladder3, name="ladder3")

            gfc1 = fc_bn_relu(ladder3, self.cs[5])
            gconv7 = fc_bn_relu(gfc1, self.fs[4] * self.fs[4] * self.cs[4])
            gconv7 = tf.reshape(gconv7, tf.pack([tf.shape(gconv7)[0], self.fs[4], self.fs[4], self.cs[4]]))

            if inputs is not None:
                gconv6 = tf.nn.relu(conv2d_t_bn(gconv7, self.cs[3], [4, 4], 2) + iconv6)
            else:
                gconv6 = conv2d_t_bn_relu(gconv7, self.cs[3], [4, 4], 2)
            gconv6 = self.combine_noise(gconv6, ladder2, name="ladder2")
            gconv5 = conv2d_t_bn_relu(gconv6, self.cs[3], [4, 4], 1)

            if inputs is not None:
                gconv4 = tf.nn.relu(conv2d_t_bn(gconv5, self.cs[2], [4, 4], 2) + iconv4)
            else:
                gconv4 = conv2d_t_bn_relu(gconv5, self.cs[2], [4, 4], 2)
            gconv4 = self.combine_noise(gconv4, ladder1, name="ladder1")
            gconv3 = conv2d_t_bn_relu(gconv4, self.cs[2], [4, 4], 1)

            if inputs is not None:
                gconv2 = tf.nn.relu(conv2d_t_bn(gconv3, self.cs[1], [4, 4], 2) + iconv2)
            else:
                gconv2 = conv2d_t_bn_relu(gconv3, self.cs[1], [4, 4], 2)
            gconv2 = self.combine_noise(gconv2, ladder0, name="ladder0")
            gconv1 = conv2d_t_bn_relu(gconv2, self.cs[1], [4, 4], 1)

            output = conv2d_t(gconv1, self.data_dims[-1], [4, 4], 2, activation_fn=tf.sigmoid)
            output = (self.dataset.range[1] - self.dataset.range[0]) * output + self.dataset.range[0]
            if inputs is not None:
                ratio = conv2d_t(gconv1, 1, [4, 4], 2, activation_fn=tf.sigmoid)
                ratio = tf.tile(ratio, (1, 1, 1, self.data_dims[-1]))
                output = tf.mul(ratio, output) + tf.mul(1 - ratio, inputs)
                return output, ratio
            else:
                return output
