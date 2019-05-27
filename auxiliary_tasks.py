import tensorflow as tf
import numpy as np 

from utils import small_convnet, fc, activ, flatten_two_dims, unflatten_first_dim, small_deconvnet, image_warp, my_deconv2d

slim = tf.contrib.slim
class FeatureExtractor(object):
    def __init__(self, policy, features_shared_with_policy, feat_dim=None, layernormalize=None,
                 scope='feature_extractor'):
        self.scope = scope
        self.features_shared_with_policy = features_shared_with_policy
        self.feat_dim = feat_dim
        self.layernormalize = layernormalize
        self.policy = policy
        self.hidsize = policy.hidsize
        self.ob_space = policy.ob_space
        self.ac_space = policy.ac_space
        self.obs = self.policy.ph_ob
        self.ob_mean = self.policy.ob_mean
        self.ob_std = self.policy.ob_std
        with tf.variable_scope(scope):
            self.last_ob = tf.placeholder(dtype=tf.int32,
                                          shape=(None, 1) + self.ob_space.shape, name='last_ob')
            self.next_ob = tf.concat([self.obs[:, 1:], self.last_ob], 1)

            if features_shared_with_policy:
                self.features = self.policy.features
                self.last_features = self.policy.get_features(self.last_ob, reuse=True)
            else:
                self.features = self.get_features(self.obs, reuse=False)
                self.last_features = self.get_features(self.last_ob, reuse=True)
            self.next_features = tf.concat([self.features[:, 1:], self.last_features], 1)

            self.ac = self.policy.ph_ac
            self.scope = scope

            self.loss = self.get_loss()

    def get_features(self, x, reuse):
        nl = tf.nn.leaky_relu
        x_has_timesteps = (x.get_shape().ndims == 5)
        if x_has_timesteps:
            sh = tf.shape(x)
            x = flatten_two_dims(x)
        with tf.variable_scope(self.scope + "_features", reuse=reuse):
            x = (tf.to_float(x) - self.ob_mean) / self.ob_std
            x = small_convnet(x, nl=nl, feat_dim=self.feat_dim, last_nl=None, layernormalize=self.layernormalize)
        if x_has_timesteps:
            x = unflatten_first_dim(x, sh)
        return x

    def get_loss(self):
        return tf.zeros((), dtype=tf.float32)

class InverseDynamics(FeatureExtractor):
    def __init__(self, policy, features_shared_with_policy, feat_dim=None, layernormalize=None):
        super(InverseDynamics, self).__init__(scope="inverse_dynamics", policy=policy,
                                              features_shared_with_policy=features_shared_with_policy,
                                              feat_dim=feat_dim, layernormalize=layernormalize)

    def get_loss(self):
        with tf.variable_scope(self.scope):
            x = tf.concat([self.features, self.next_features], 2)
            sh = tf.shape(x)
            x = flatten_two_dims(x)
            x = fc(x, units=self.policy.hidsize, activation=activ)
            x = fc(x, units=self.ac_space.n, activation=None)
            param = unflatten_first_dim(x, sh)
            idfpd = self.policy.ac_pdtype.pdfromflat(param)
            return idfpd.neglogp(self.ac)


class OpticalFlowFeatureExtractor(object):
    def __init__(self, policy, FICM_type='flowC', fix_features=False, scope='flow_feature_extractor'):
        print('Using OpticalFlow FeatureExtractor.')

        self.scope = scope
        self.policy = policy
        self.hidsize = policy.hidsize
        self.ob_space = policy.ob_space
        self.ac_space = policy.ac_space
        self.obs = self.policy.ph_ob

        # Since Optical flow input range in [0, 1]
        self.ob_mean = self.policy.ob_mean / 255.0 
        self.ob_std = self.policy.ob_std / 255.0

        with tf.variable_scope(scope):
            self.last_ob = tf.placeholder(dtype=tf.int32,
                                          shape=(None, 1) + self.ob_space.shape, name='last_ob')
            self.next_ob = tf.concat([self.obs[:, 1:], self.last_ob], 1)

            # Get the last frame -> (?, ?, 84, 84, 1), since optical flow need only two frames.
            # obs -> (batch, 128, 84, 84, 1)
            # last_ob -> (batch, 1, 84, 84, 1)
            obs = self.obs[:, :, :, :, -1:] 
            last_ob = self.last_ob[:, :, :, :, -1:]

            self.obs_sh = tf.shape(obs)
            self.last_ob_sh = tf.shape(last_ob)

            self.h, self.w = obs.get_shape().as_list()[2:4]
            self.ac = self.policy.ph_ac

            # Divide 255.0, let the input observation range in [0, 1] (Take as warping input)
            obs = tf.divide(tf.to_float(obs), 255.0)
            last_ob = tf.divide(tf.to_float(last_ob), 255.0)
            next_ob = tf.concat([obs[:, 1:], last_ob], axis=1)

            obs = flatten_two_dims(obs)
            last_ob = flatten_two_dims(last_ob)
            next_ob = flatten_two_dims(next_ob)

            self.obs_warped_input = obs
            self.last_ob_warped_input = last_ob
            self.next_ob_warped_input = next_ob

            # Input for neural network input with mean-zero values in [-1, 1] (Take as network input)
            obs_normalized = (self.obs_warped_input - self.ob_mean[:, :, 2:3]) / self.ob_std
            last_ob_normalized = (self.last_ob_warped_input - self.ob_mean[:, :, 3:]) / self.ob_std
            next_ob_normalized = (self.next_ob_warped_input - self.ob_mean[:, :, 3:]) / self.ob_std

            print('FICM type: ', FICM_type)
            # Get features from input observation
            # features_l contains features of 128 observations
            # features_r contains feautres of only 1 observation (last observation)
            if FICM_type == 'flowC':
                with tf.variable_scope(self.scope + "_features_C", reuse=False):
                    features_l = self.get_flowC_features(obs_normalized, fix_features)

                with tf.variable_scope(self.scope + "_features_C", reuse=True):
                    features_r = self.get_flowC_features(last_ob_normalized, fix_features)

                # Need to unflatten to find the correct position to concat.
                features_l_unflat = [unflatten_first_dim(f, self.obs_sh) for f in features_l]
                features_r_unflat = [unflatten_first_dim(f, self.last_ob_sh) for f in features_r]

                features_r_concat = []
                for f_l, f_r in zip(features_l_unflat, features_r_unflat):
                    features_r_concat.append(tf.concat([f_l[:, 1:], f_r], axis=1))

                features_l_flat = [flatten_two_dims(f) for f in features_l_unflat]
                features_r_flat = [flatten_two_dims(f) for f in features_r_concat]

                self.conv3_l = features_l_flat[2]
                self.conv3_r = features_r_flat[2]
                self.conv2_l = features_l_flat[1]
                self.conv2_r = features_r_flat[1]

                with tf.variable_scope(self.scope + "_flowC", reuse=False):
                    flow_fw, corr_fw = self.flowC(self.conv3_l, self.conv3_r, self.conv2_l)
                    
                with tf.variable_scope(self.scope + "_flowC", reuse=True):
                    flow_bw, corr_bw = self.flowC(self.conv3_r, self.conv3_l, self.conv2_r)

                # For forward dynamics (We don't use these at this time.)
                # self.features = unflatten_first_dim(self.conv3_l, self.obs_sh)
                # self.next_features = unflatten_first_dim(self.conv3_r, self.obs_sh)

            elif FICM_type == 'flowS':
                obs_stack_fw = tf.concat([obs_normalized, next_ob_normalized], axis=3)
                obs_stack_bw = tf.concat([next_ob_normalized, obs_normalized], axis=3)

                with tf.variable_scope(self.scope + "_features_S", reuse=False):
                    features_fw = self.get_flowS_features(obs_stack_fw, fix_features)
                    flow_fw = self.flowS(features_fw[0], features_fw[1], features_fw[2])

                with tf.variable_scope(self.scope + "_features_S", reuse=True):
                    features_bw = self.get_flowS_features(obs_stack_bw, fix_features)
                    flow_bw = self.flowS(features_bw[0], features_bw[1], features_bw[2])

            ## Optical flow for training flow module
            self.flow_fw_up = tf.image.resize_bilinear(flow_fw, [self.h, self.w]) * 5.0
            self.flow_bw_up = tf.image.resize_bilinear(flow_bw, [self.h, self.w]) * 5.0

            self.alpha = 0.45
            self.beta = 255

            self.loss, self.pred_error = self.get_loss(alpha=self.alpha, beta=self.beta, epsilon=0.001)

    def get_loss(self, alpha, beta, epsilon):
        # Return: [flow_loss, pred_error]
        # flow_loss is used for training flow network.
        # pred_error is used as the flow-based intrinsic signal.

        _obs = image_warp(self.next_ob_warped_input, self.flow_fw_up)
        _next_ob = image_warp(self.obs_warped_input, self.flow_bw_up)
        
        fw_diff_ob = tf.reshape((self.obs_warped_input - _obs), self.obs_sh) * beta
        bw_diff_ob = tf.reshape((self.next_ob_warped_input - _next_ob), self.obs_sh) * beta

        fw_loss_ob = tf.pow(tf.square(fw_diff_ob) + tf.square(epsilon), alpha)
        bw_loss_ob = tf.pow(tf.square(bw_diff_ob) + tf.square(epsilon), alpha)

        pred_error = tf.reduce_mean(fw_loss_ob, axis=[2, 3, 4]) + tf.reduce_mean(bw_loss_ob, axis=[2, 3, 4])
        flow_loss = tf.reduce_mean(fw_loss_ob + bw_loss_ob)

        return flow_loss, pred_error

    def get_flowC_features(self, x, fix_features=False):
        elu = tf.nn.elu

        l1_x = slim.conv2d(x, 32, [4, 4], activation_fn=elu, stride=2, scope='l1')
        l2_x = slim.conv2d(l1_x, 64, [4, 4], activation_fn=elu, stride=2, scope='l2')
        l3_x = slim.conv2d(l2_x, 96, [3, 3], activation_fn=elu, stride=2, scope='l3')
        
        if fix_features == True:
            l1_x = tf.stop_gradient(l1_x)
            l2_x = tf.stop_gradient(l2_x)
            l3_x = tf.stop_gradient(l3_x)

        return l1_x, l2_x, l3_x

    def get_flowS_features(self, x, fix_features=False):
        elu = tf.nn.elu
    
        l1_x = slim.conv2d(x, 32, [4, 4], activation_fn=elu, stride=2, scope='l1')
        l2_x = slim.conv2d(l1_x, 64, [4, 4], activation_fn=elu, stride=2, scope='l2')
        l3_x = slim.conv2d(l2_x, 64, [3, 3], activation_fn=elu, stride=2, scope='l3')
        l4_x = slim.conv2d(l3_x, 96, [3, 3], activation_fn=elu, stride=2, scope='l4')

        if fix_features == True:
            l2_x = tf.stop_gradient(l2_x)
            l3_x = tf.stop_gradient(l3_x)
            l4_x = tf.stop_gradient(l4_x)

        return l2_x, l3_x, l4_x

    def flowC(self, conv3_l, conv3_r, conv2_l):
        from correlation_layer.src.correlation import correlation

        elu = tf.nn.elu
    
        ### Correlation 
        corr = correlation(conv3_l, conv3_r, 1, 3, 1, 2, 3) # kernel_size, max_displacement, stride_1, stride_2, padding
        corr_relu = tf.nn.leaky_relu(corr)
        
        conv_redir = slim.conv2d(conv3_l, 32, 1, activation_fn=elu, stride=1, scope='conv_redir')
        ###

        concat3 = tf.concat([conv_redir, corr_relu], axis=3)
        conv3_1 = slim.conv2d(concat3, 64, [3, 3], activation_fn=elu, stride=1, scope='cpnv3_1')
        
        conv4 = slim.conv2d(conv3_1, 96, [3, 3], activation_fn=elu, stride=2, scope='conv4')
        conv4_1 = slim.conv2d(conv4, 96, 3, stride=1, scope='conv4_1')
        
        dl2_x = elu(my_deconv2d(conv4_1, 64, [3, 3], stride=2, out_shape=[11, 11], c_i=96, name='dl2'))
        concat2 = tf.concat([conv3_l, dl2_x], axis=3)   

        dl1_x = elu(my_deconv2d(concat2, 64, [3, 3], stride=2, out_shape=[21, 21], c_i=160, name='dl1')) # 32
        concat1 = tf.concat([conv2_l, dl1_x], axis=3)   
        
        flow = slim.conv2d(concat1, 2, [3, 3], activation_fn=None, stride=1)
        
        return flow, corr

    def flowS(self, l2_x, l3_x, l4_x):
        dl2_x = tf.nn.elu(my_deconv2d(l4_x, 64, [3, 3], stride=2, out_shape=[11, 11], c_i=96, name='dl2'))
        concat2 = tf.concat([l3_x, dl2_x], axis=3)

        dl1_x = tf.nn.elu(my_deconv2d(concat2, 32, [3, 3], stride=2, out_shape=[21, 21], c_i=128, name='dl1'))
        concat1 = tf.concat([l2_x, dl1_x], axis=3)

        flow = slim.conv2d(concat1, 2, [3, 3], activation_fn=None, stride=1)

        return flow

class VAE(FeatureExtractor):
    def __init__(self, policy, features_shared_with_policy, feat_dim=None, layernormalize=False, spherical_obs=False):
        assert not layernormalize, "VAE features should already have reasonable size, no need to layer normalize them"
        self.spherical_obs = spherical_obs
        super(VAE, self).__init__(scope="vae", policy=policy,
                                  features_shared_with_policy=features_shared_with_policy,
                                  feat_dim=feat_dim, layernormalize=False)
        self.features = tf.split(self.features, 2, -1)[0]  # use mean only for features exposed to the dynamics
        self.next_features = tf.split(self.next_features, 2, -1)[0]

    def get_features(self, x, reuse):
        nl = tf.nn.leaky_relu
        x_has_timesteps = (x.get_shape().ndims == 5)
        if x_has_timesteps:
            sh = tf.shape(x)
            x = flatten_two_dims(x)
        with tf.variable_scope(self.scope + "_features", reuse=reuse):
            x = (tf.to_float(x) - self.ob_mean) / self.ob_std
            x = small_convnet(x, nl=nl, feat_dim=2 * self.feat_dim, last_nl=None, layernormalize=False)
        if x_has_timesteps:
            x = unflatten_first_dim(x, sh)
        return x

    def get_loss(self):
        with tf.variable_scope(self.scope):
            posterior_mean, posterior_scale = tf.split(self.features, 2, -1)
            posterior_scale = tf.nn.softplus(posterior_scale)
            posterior_distribution = tf.distributions.Normal(loc=posterior_mean, scale=posterior_scale)

            sh = tf.shape(posterior_mean)
            prior = tf.distributions.Normal(loc=tf.zeros(sh), scale=tf.ones(sh))

            posterior_kl = tf.distributions.kl_divergence(posterior_distribution, prior)

            posterior_kl = tf.reduce_sum(posterior_kl, [-1])
            assert posterior_kl.get_shape().ndims == 2

            posterior_sample = posterior_distribution.sample()
            reconstruction_distribution = self.decoder(posterior_sample)
            norm_obs = self.add_noise_and_normalize(self.obs)
            reconstruction_likelihood = reconstruction_distribution.log_prob(norm_obs)
            assert reconstruction_likelihood.get_shape().as_list()[2:] == [84, 84, 4]
            reconstruction_likelihood = tf.reduce_sum(reconstruction_likelihood, [2, 3, 4])

            likelihood_lower_bound = reconstruction_likelihood - posterior_kl
            return - likelihood_lower_bound

    def add_noise_and_normalize(self, x):
        x = tf.to_float(x) + tf.random_uniform(shape=tf.shape(x), minval=0., maxval=1.)
        x = (x - self.ob_mean) / self.ob_std
        return x

    def decoder(self, z):
        nl = tf.nn.leaky_relu
        z_has_timesteps = (z.get_shape().ndims == 3)
        if z_has_timesteps:
            sh = tf.shape(z)
            z = flatten_two_dims(z)
        with tf.variable_scope(self.scope + "decoder"):
            z = small_deconvnet(z, nl=nl, ch=4 if self.spherical_obs else 8, positional_bias=True)
            if z_has_timesteps:
                z = unflatten_first_dim(z, sh)
            if self.spherical_obs:
                scale = tf.get_variable(name="scale", shape=(), dtype=tf.float32,
                                        initializer=tf.ones_initializer())
                scale = tf.maximum(scale, -4.)
                scale = tf.nn.softplus(scale)
                scale = scale * tf.ones_like(z)
            else:
                z, scale = tf.split(z, 2, -1)
                scale = tf.nn.softplus(scale)
            # scale = tf.Print(scale, [scale])
            return tf.distributions.Normal(loc=z, scale=scale)


class JustPixels(FeatureExtractor):
    def __init__(self, policy, features_shared_with_policy, feat_dim=None, layernormalize=None,
                 scope='just_pixels'):
        assert not layernormalize
        assert not features_shared_with_policy
        super(JustPixels, self).__init__(scope=scope, policy=policy,
                                         features_shared_with_policy=False,
                                         feat_dim=None, layernormalize=None)

    def get_features(self, x, reuse):
        with tf.variable_scope(self.scope + "_features", reuse=reuse):
            x = (tf.to_float(x) - self.ob_mean) / self.ob_std
        return x

    def get_loss(self):
        return tf.zeros((), dtype=tf.float32)
