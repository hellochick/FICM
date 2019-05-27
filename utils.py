import multiprocessing
import os
import platform
from functools import partial

import numpy as np
import tensorflow as tf
from baselines.common.tf_util import normc_initializer
from mpi4py import MPI

def make_var(name, shape):
        '''Creates a new TensorFlow variable.'''
        return tf.get_variable(name, shape, trainable=True)

def my_deconv2d(input, c_o, k_size, stride, out_shape, c_i, name):
        bs = tf.shape(input)[0]

        output_shape = [bs, out_shape[0], out_shape[1], c_o]
    
        deconv = lambda i, k: tf.nn.conv2d_transpose(i, k, output_shape, [1, stride, stride, 1])
        with tf.variable_scope(name) as scope:
            kernel = make_var('weights', shape=[k_size[0], k_size[1], c_o, c_i])
            output = deconv(input, kernel)

            return output

def image_warp(im, flow):
    """Performs a backward warp of an image using the predicted flow.
    Args:
        im: Batch of images. [num_batch, height, width, channels]
        flow: Batch of flow vectors. [num_batch, height, width, 2]
    Returns:
        warped: transformed image of the same shape as the input image.
    """
    with tf.variable_scope('image_warp'):

        num_batch, height, width, channels = tf.unstack(tf.shape(im))
        max_x = tf.cast(width - 1, 'int32')
        max_y = tf.cast(height - 1, 'int32')
        zero = tf.zeros([], dtype='int32')

        # We have to flatten our tensors to vectorize the interpolation
        im_flat = tf.reshape(im, [-1, channels])
        flow_flat = tf.reshape(flow, [-1, 2])

        # Floor the flow, as the final indices are integers
        # The fractional part is used to control the bilinear interpolation.
        flow_floor = tf.to_int32(tf.floor(flow_flat))
        bilinear_weights = flow_flat - tf.floor(flow_flat)

        # Construct base indices which are displaced with the flow
        pos_x = tf.tile(tf.range(width), [height * num_batch])
        grid_y = tf.tile(tf.expand_dims(tf.range(height), 1), [1, width])
        pos_y = tf.tile(tf.reshape(grid_y, [-1]), [num_batch])

        x = flow_floor[:, 0]
        y = flow_floor[:, 1]
        xw = bilinear_weights[:, 0]
        yw = bilinear_weights[:, 1]

        # Compute interpolation weights for 4 adjacent pixels
        # expand to num_batch * height * width x 1 for broadcasting in add_n below
        wa = tf.expand_dims((1 - xw) * (1 - yw), 1) # top left pixel
        wb = tf.expand_dims((1 - xw) * yw, 1) # bottom left pixel
        wc = tf.expand_dims(xw * (1 - yw), 1) # top right pixel
        wd = tf.expand_dims(xw * yw, 1) # bottom right pixel

        x0 = pos_x + x
        x1 = x0 + 1
        y0 = pos_y + y
        y1 = y0 + 1

        x0 = tf.clip_by_value(x0, zero, max_x)
        x1 = tf.clip_by_value(x1, zero, max_x)
        y0 = tf.clip_by_value(y0, zero, max_y)
        y1 = tf.clip_by_value(y1, zero, max_y)

        dim1 = width * height
        batch_offsets = tf.range(num_batch) * dim1
        base_grid = tf.tile(tf.expand_dims(batch_offsets, 1), [1, dim1])
        base = tf.reshape(base_grid, [-1])

        base_y0 = base + y0 * width
        base_y1 = base + y1 * width
        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        Ia = tf.gather(im_flat, idx_a)
        Ib = tf.gather(im_flat, idx_b)
        Ic = tf.gather(im_flat, idx_c)
        Id = tf.gather(im_flat, idx_d)

        warped_flat = tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])
        warped = tf.reshape(warped_flat, [num_batch, height, width, channels])

        return warped
        
def bcast_tf_vars_from_root(sess, vars):
    """
    Send the root node's parameters to every worker.

    Arguments:
      sess: the TensorFlow session.
      vars: all parameter variables including optimizer's
    """
    rank = MPI.COMM_WORLD.Get_rank()
    for var in vars:
        if rank == 0:
            MPI.COMM_WORLD.bcast(sess.run(var))
        else:
            sess.run(tf.assign(var, MPI.COMM_WORLD.bcast(None)))


def get_mean_and_std(array):
    comm = MPI.COMM_WORLD
    task_id, num_tasks = comm.Get_rank(), comm.Get_size()
    local_mean = np.array(np.mean(array))
    sum_of_means = np.zeros((), dtype=np.float32)
    comm.Allreduce(local_mean, sum_of_means, op=MPI.SUM)
    mean = sum_of_means / num_tasks

    n_array = array - mean
    sqs = n_array ** 2
    local_mean = np.array(np.mean(sqs))
    sum_of_means = np.zeros((), dtype=np.float32)
    comm.Allreduce(local_mean, sum_of_means, op=MPI.SUM)
    var = sum_of_means / num_tasks
    std = var ** 0.5
    return mean, std


def guess_available_gpus(n_gpus=None):
    if n_gpus is not None:
        return list(range(n_gpus))
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        cuda_visible_divices = os.environ['CUDA_VISIBLE_DEVICES']
        cuda_visible_divices = cuda_visible_divices.split(',')
        return [int(n) for n in cuda_visible_divices]
    nvidia_dir = '/proc/driver/nvidia/gpus/'
    if os.path.exists(nvidia_dir):
        n_gpus = len(os.listdir(nvidia_dir))
        return list(range(n_gpus))
    raise Exception("Couldn't guess the available gpus on this machine")


def setup_mpi_gpus():
    """
    Set CUDA_VISIBLE_DEVICES using MPI.
    """
    available_gpus = guess_available_gpus()

    node_id = platform.node()
    nodes_ordered_by_rank = MPI.COMM_WORLD.allgather(node_id)
    processes_outranked_on_this_node = [n for n in nodes_ordered_by_rank[:MPI.COMM_WORLD.Get_rank()] if n == node_id]
    local_rank = len(processes_outranked_on_this_node)

    print('CUDA Device: ', str(available_gpus[local_rank]))
    os.environ['CUDA_VISIBLE_DEVICES'] = str(available_gpus[local_rank])


def guess_available_cpus():
    return int(multiprocessing.cpu_count())


def setup_tensorflow_session():
    num_cpu = guess_available_cpus()

    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=num_cpu,
        intra_op_parallelism_threads=num_cpu
    )
    return tf.Session(config=tf_config)


def random_agent_ob_mean_std(env, nsteps=10000):
    ob = np.asarray(env.reset())
    if MPI.COMM_WORLD.Get_rank() == 0:
        obs = [ob]
        for _ in range(nsteps):
            ac = env.action_space.sample()
            ob, _, done, _ = env.step(ac)
            if done:
                ob = env.reset()
            obs.append(np.asarray(ob))
        print(np.asarray(ob).shape)
        mean = np.mean(obs, 0).astype(np.float32)
        std = np.std(obs, 0).mean().astype(np.float32)
    else:
        mean = np.empty(shape=ob.shape, dtype=np.float32)
        std = np.empty(shape=(), dtype=np.float32)
    MPI.COMM_WORLD.Bcast(mean, root=0)
    MPI.COMM_WORLD.Bcast(std, root=0)
    return mean, std


def layernorm(x):
    m, v = tf.nn.moments(x, -1, keep_dims=True)
    return (x - m) / (tf.sqrt(v) + 1e-8)


getsess = tf.get_default_session

fc = partial(tf.layers.dense, kernel_initializer=normc_initializer(1.))
activ = tf.nn.relu


def flatten_two_dims(x):
    return tf.reshape(x, [-1] + x.get_shape().as_list()[2:])


def unflatten_first_dim(x, sh):
    return tf.reshape(x, [sh[0], sh[1]] + x.get_shape().as_list()[1:])


def add_pos_bias(x):
    with tf.variable_scope(name_or_scope=None, default_name="pos_bias"):
        b = tf.get_variable(name="pos_bias", shape=[1] + x.get_shape().as_list()[1:], dtype=tf.float32,
                            initializer=tf.zeros_initializer())
        return x + b


def small_convnet(x, nl, feat_dim, last_nl, layernormalize, batchnorm=False):
    bn = tf.layers.batch_normalization if batchnorm else lambda x: x
    x = bn(tf.layers.conv2d(x, filters=32, kernel_size=8, strides=(4, 4), activation=nl))
    x = bn(tf.layers.conv2d(x, filters=64, kernel_size=4, strides=(2, 2), activation=nl))
    x = bn(tf.layers.conv2d(x, filters=64, kernel_size=3, strides=(1, 1), activation=nl))
    x = tf.reshape(x, (-1, np.prod(x.get_shape().as_list()[1:])))
    x = bn(fc(x, units=feat_dim, activation=None))
    if last_nl is not None:
        x = last_nl(x)
    if layernormalize:
        x = layernorm(x)
    return x


def small_deconvnet(z, nl, ch, positional_bias):
    sh = (8, 8, 64)
    z = fc(z, np.prod(sh), activation=nl)
    z = tf.reshape(z, (-1, *sh))
    z = tf.layers.conv2d_transpose(z, 128, kernel_size=4, strides=(2, 2), activation=nl, padding='same')
    assert z.get_shape().as_list()[1:3] == [16, 16]
    z = tf.layers.conv2d_transpose(z, 64, kernel_size=8, strides=(2, 2), activation=nl, padding='same')
    assert z.get_shape().as_list()[1:3] == [32, 32]
    z = tf.layers.conv2d_transpose(z, ch, kernel_size=8, strides=(3, 3), activation=None, padding='same')
    assert z.get_shape().as_list()[1:3] == [96, 96]
    z = z[:, 6:-6, 6:-6]
    assert z.get_shape().as_list()[1:3] == [84, 84]
    if positional_bias:
        z = add_pos_bias(z)
    return z


def unet(x, nl, feat_dim, cond, batchnorm=False):
    bn = tf.layers.batch_normalization if batchnorm else lambda x: x
    layers = []
    x = tf.pad(x, [[0, 0], [6, 6], [6, 6], [0, 0]])
    x = bn(tf.layers.conv2d(cond(x), filters=32, kernel_size=8, strides=(3, 3), activation=nl, padding='same'))
    assert x.get_shape().as_list()[1:3] == [32, 32]
    layers.append(x)
    x = bn(tf.layers.conv2d(cond(x), filters=64, kernel_size=8, strides=(2, 2), activation=nl, padding='same'))
    layers.append(x)
    assert x.get_shape().as_list()[1:3] == [16, 16]
    x = bn(tf.layers.conv2d(cond(x), filters=64, kernel_size=4, strides=(2, 2), activation=nl, padding='same'))
    layers.append(x)
    assert x.get_shape().as_list()[1:3] == [8, 8]

    x = tf.reshape(x, (-1, np.prod(x.get_shape().as_list()[1:])))
    x = fc(cond(x), units=feat_dim, activation=nl)

    def residual(x):
        res = bn(tf.layers.dense(cond(x), feat_dim, activation=tf.nn.leaky_relu))
        res = tf.layers.dense(cond(res), feat_dim, activation=None)
        return x + res

    for _ in range(4):
        x = residual(x)

    sh = (8, 8, 64)
    x = fc(cond(x), np.prod(sh), activation=nl)
    x = tf.reshape(x, (-1, *sh))
    x += layers.pop()
    x = bn(tf.layers.conv2d_transpose(cond(x), 64, kernel_size=4, strides=(2, 2), activation=nl, padding='same'))
    assert x.get_shape().as_list()[1:3] == [16, 16]
    x += layers.pop()
    x = bn(tf.layers.conv2d_transpose(cond(x), 32, kernel_size=8, strides=(2, 2), activation=nl, padding='same'))
    assert x.get_shape().as_list()[1:3] == [32, 32]
    x += layers.pop()
    x = tf.layers.conv2d_transpose(cond(x), 4, kernel_size=8, strides=(3, 3), activation=None, padding='same')
    assert x.get_shape().as_list()[1:3] == [96, 96]
    x = x[:, 6:-6, 6:-6]
    assert x.get_shape().as_list()[1:3] == [84, 84]
    assert layers == []
    return x


def tile_images(array, n_cols=None, max_images=None, div=1):
    if max_images is not None:
        array = array[:max_images]
    if len(array.shape) == 4 and array.shape[3] == 1:
        array = array[:, :, :, 0]
    assert len(array.shape) in [3, 4], "wrong number of dimensions - shape {}".format(array.shape)
    if len(array.shape) == 4:
        assert array.shape[3] == 3, "wrong number of channels- shape {}".format(array.shape)
    if n_cols is None:
        n_cols = max(int(np.sqrt(array.shape[0])) // div * div, div)
    n_rows = int(np.ceil(float(array.shape[0]) / n_cols))

    def cell(i, j):
        ind = i * n_cols + j
        return array[ind] if ind < array.shape[0] else np.zeros(array[0].shape)

    def row(i):
        return np.concatenate([cell(i, j) for j in range(n_cols)], axis=1)

    return np.concatenate([row(i) for i in range(n_rows)], axis=0)

