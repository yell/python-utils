import os
from abc import ABCMeta, abstractmethod
from StringIO import StringIO

import matplotlib; matplotlib.use('Agg'); from matplotlib.pyplot import imsave
import numpy as np
import tensorflow as tf

from .. import AttrDict, set_readonly_property


class TensorFlowModel(object):
    """
    A utility class encapsulating basic TensorFlow infrastructure
    for executing computation, saving/restoring, and visualization.

    Parameters
    ----------
    global_step : {None, int}
        If provided, initialize global step with this value.
        Also, if provided, load checkpoint with specified value of
        global step if `load=True`.

    References
    ----------
    * Logging to tensorboard without tensorflow operations.
      https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
    """
    __metaclass__ = ABCMeta

    def __init__(self,
                 model_dirpath='tf_model/',
                 load_best=False,
                 load_latest=False,
                 global_step=None,
                 lazy=False,
                 tf_session=None,
                 tf_graph=None,
                 tf_best_saver_kwargs=None,
                 tf_latest_saver_kwargs=None):
        super(TensorFlowModel, self).__init__()

        self.model_dirpath = os.path.expanduser(model_dirpath)

        assert (load_best and load_latest) == False
        self.load_best = load_best
        self.load_latest = load_latest
        self.global_step_init = global_step
        self.lazy = lazy

        tf_ = AttrDict()
        tf_.best_saver_kwargs = tf_best_saver_kwargs or {}
        tf_.best_saver_kwargs.setdefault('name', 'best_save')
        tf_.best_saver_kwargs.setdefault('save_relative_paths', True)
        tf_.latest_saver_kwargs = tf_latest_saver_kwargs or {}
        tf_.latest_saver_kwargs.setdefault('name', 'latest_save')
        tf_.latest_saver_kwargs.setdefault('save_relative_paths', True)

        # define session
        tf_.session = tf_session or tf.get_default_session() or tf.Session()

        # define graph, will be passed to (training)
        # writer for visualization purposes
        tf_.graph = tf_graph or tf_.session.graph

        # define global step
        tf_.global_step = tf.train.get_or_create_global_step(tf_.graph)

        # gather all TensorFlow entities into the readonly property 'tf'
        set_readonly_property(self.__class__, 'tf', tf_)

        # build TensorFlow graph for the model (by the end of this method call,
        # at least all the variables one wants to initialize and save automatically
        # should be defined
        self._make_tf_model()

        self._finalized = False
        if not self.lazy:
            self.finalize_tf_model()

    @abstractmethod
    def _make_tf_model(self):
        pass

    def _check_global_step(self, global_step):
        return self.global_step if global_step is None else global_step

    def _check_train_writer(self):
        if not self.tf.train_writer:
            self.tf.train_writer = tf.summary.FileWriter(self.train_summary_dirpath,
                                                         self.tf.graph)
        return self.tf.train_writer

    def _check_val_writer(self):
        if not self.tf.val_writer:
            self.tf.val_writer = tf.summary.FileWriter(self.val_summary_dirpath)
        return self.tf.val_writer

    @staticmethod
    def load_tf_model(checkpoint_dirpath, tf_session, tf_saver=None, is_best=False,
                      global_step=None, *tf_restore_args, **tf_restore_kwargs):
        ckpt_dirpath = os.path.join(checkpoint_dirpath, 'best/' if is_best else 'latest/')
        if not os.path.isdir(ckpt_dirpath):
            raise IOError("directory '{}' does not exist".format(ckpt_dirpath))

        ckpt_state = tf.train.get_checkpoint_state(ckpt_dirpath)
        if ckpt_state is None:
            raise IOError("no checkpoints to restore from in '{}'".format(ckpt_dirpath))

        # load checkpoint with specified global step
        if global_step is not None:
            for ckpt_path in ckpt_state.all_model_checkpoint_paths:
                if ckpt_path.endswith('-{}'.format(global_step)):
                    load_path = ckpt_path
                    break
            else:
                raise IOError('no checkpoint with `global_step={}`'.format(global_step))

        # otherwise load latest checkpoint
        else:
            load_path = ckpt_state.model_checkpoint_path  # == tf.train.latest_checkpoint(dirpath)

        if tf_saver is None:
            tf_saver = tf.train.import_meta_graph(load_path + '.meta')

        return tf_saver.restore(tf_session, load_path, *tf_restore_args, **tf_restore_kwargs)

    @property
    def global_step(self):
        return self.run(self.tf.global_step)  # == tf.train.global_step(self.tf.session,
                                              #                         self.tf.global_step)

    @global_step.setter
    def global_step(self, value):
        value = tf.constant(value, dtype=tf.int64)
        self.run(self.tf.global_step.assign(value))

    def finalize_tf_model(self):
        if self._finalized:
            return

        # initialize savers
        self.tf.best_saver = tf.train.Saver(**self.tf.best_saver_kwargs)
        self.tf.latest_saver = tf.train.Saver(**self.tf.latest_saver_kwargs)

        # load model if necessary
        if self.load_best or self.load_latest:
            TensorFlowModel.load_tf_model(self.model_dirpath,
                                          tf_session=self.tf.session,
                                          tf_saver=self.tf.best_saver if self.load_best else self.tf.latest_saver,
                                          is_best=self.load_best,
                                          global_step=self.global_step_init)

            # initialize uninitialized variables if any
            global_vars = tf.global_variables()
            is_initialized = self.run(map(tf.is_variable_initialized, global_vars))
            not_initialized_vars = [v for (v, b) in zip(global_vars, is_initialized) if not b]
            if len(not_initialized_vars):
                self.run(tf.variables_initializer(not_initialized_vars))
        else:
            # initialize variables
            init_op = tf.global_variables_initializer()
            self.run(init_op)

        # set global step if needed
        if self.global_step_init is not None:
            self.global_step = self.global_step_init

        # set summary attributes
        self.train_summary_dirpath = os.path.join(self.model_dirpath, 'train_logs')
        self.val_summary_dirpath = os.path.join(self.model_dirpath, 'val_logs')
        self.tf.merged_summaries = tf.summary.merge_all()
        self.tf.train_writer = None
        self.tf.val_writer = None

        # update flag
        self._finalized = True

    def run(self, *args, **kwargs):
        return self.tf.session.run(*args, **kwargs)

    def save_tf_model(self, checkpoint_prefix='model.ckpt', is_best=False, global_step=None,
                      write_meta_graph=False, *tf_save_args, **tf_save_kwargs):
        self._check_train_writer()  # save graph even if no summaries defined
                                    # or called
        ckpt_filepath = os.path.join(self.model_dirpath,
                                     'best' if is_best else 'latest',
                                     checkpoint_prefix)
        global_step = self._check_global_step(global_step)
        saver = self.tf.best_saver if is_best else self.tf.latest_saver
        return saver.save(self.tf.session, ckpt_filepath,
                          global_step=global_step, write_meta_graph=write_meta_graph,
                          *tf_save_args, **tf_save_kwargs)

    def run_summary(self, summary_ops=None, global_step=None, train=True):
        if summary_ops is None:
            summary_ops = self.tf.merged_summaries

        global_step = self._check_global_step(global_step)
        writer = self._check_train_writer() if train else self._check_val_writer()

        summary_str = self.run(summary_ops)
        writer.add_summary(summary_str, global_step)
        writer.flush()

    def log_scalars(self, tags, values, global_step=None, train=True):
        tags = np.atleast_1d(tags)
        values = np.atleast_1d(values)
        assert len(tags) == len(values)

        global_step = self._check_global_step(global_step)
        writer = self._check_train_writer() if train else self._check_val_writer()

        summary_value = [tf.Summary.Value(tag=t, simple_value=v) for t, v in zip(tags, values)]
        summary_str = tf.Summary(value=summary_value)
        writer.add_summary(summary_str, global_step)
        writer.flush()

    def log_images(self, tags, images, global_step=None, train=True):
        """
        Parameters
        ----------
        images : iterable of (H, W) or (H, W, C: {3, 4}) array-like
            Batch of images to visualize.
        """
        tags = np.atleast_1d(tags)
        assert len(tags) == len(images)

        global_step = self._check_global_step(global_step)
        writer = self._check_train_writer() if train else self._check_val_writer()

        summary_values = []
        for tag, img in zip(tags, images):
            img = np.asarray(img)
            s = StringIO()
            imsave(s, img, format='png')
            im_summary = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                          height=img.shape[0],
                                          width=img.shape[1])
            summary_value = tf.Summary.Value(tag=tag, image=im_summary)
            summary_values.append(summary_value)

        summary_str = tf.Summary(value=summary_values)
        writer.add_summary(summary_str, global_step)
        writer.flush()

    def log_hist(self, tag, values, global_step=None, bins=1000, train=True):
        # compute histogram
        values = np.asarray(values)
        counts, bin_edges = np.histogram(values, bins=bins)

        assert len(bin_edges) == bins + 1
        # TensorFlow requires len(bin_edges) == bins, where the first bin by default
        # goes from -DBL_MAX to bin_edges[1], thus we drop the first edge. See:
        # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto#L30
        bin_edges = bin_edges[1:]

        # fill proto-file
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values ** 2))
        hist.bucket.extend(counts)
        hist.bucket_limit.extend(bin_edges)

        # check other arguments and log histogram
        global_step = self._check_global_step(global_step)
        writer = self._check_train_writer() if train else self._check_val_writer()

        summary_value = tf.Summary.Value(tag=tag, histo=hist)
        summary_str = tf.Summary(value=[summary_value])
        writer.add_summary(summary_str, global_step)
        writer.flush()
