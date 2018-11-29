import os
from shutil import rmtree

import tensorflow as tf
from numpy.testing import assert_almost_equal

from tf_model import TensorFlowModel


class TestModel(TensorFlowModel):
    def __init__(self, *args, **kwargs):
        super(TestModel, self).__init__(*args, **kwargs)

    def _make_tf_model(self):
        self.x = tf.get_variable('x', initializer=tf.constant(0))
        self.y = tf.get_variable('y', initializer=tf.constant(0.))
        self.z = tf.get_variable('z', initializer=tf.constant(0.))
        self.op = self.x.assign_add(tf.constant(1))


class TestModelWithSummary(TestModel):
    def __init__(self, *args, **kwargs):
        super(TestModelWithSummary, self).__init__(*args, **kwargs)

    def _make_tf_model(self):
        super(TestModelWithSummary, self)._make_tf_model()
        tf.summary.scalar(self.x.name, self.x)


class TestTensorFlowModel(tf.test.TestCase):
    def cleanup(self):
        for d in ('test_model_1/', 'test_model_2/'):
            try:
                rmtree(d)
            except OSError:
                pass

    def test_vars_and_ops(self):
        m = TestModel()

        assert m.global_step == 0
        assert m.run(m.tf.global_step) == 0
        assert m.run(m.x) == 0
        assert_almost_equal(0., m.run(m.y))
        assert_almost_equal(0., m.run(m.z))

        m.run(m.op)
        assert m.run(m.x) == 1

    def test_global_step(self):
        with self.test_session() as sess:
            global_step = tf.train.get_or_create_global_step(sess.graph)
            m = TestModel()
            assert m.tf.global_step is global_step

            m.global_step = 1
            assert m.global_step == 1
            m.global_step = 0
            assert m.global_step == 0
            m.global_step += 10
            assert m.global_step == 10

    def test_summaries_not_created(self):
        m = TestModel()

        assert m.tf.merged_summaries is None
        assert m.tf.train_writer is None
        assert m.tf.val_writer is None

        assert not os.path.isdir(m.train_summary_dirpath)
        assert not os.path.isdir(m.val_summary_dirpath)

    def test_summaries_created(self):
        m = TestModelWithSummary(model_dirpath='test_model_1')

        assert m.tf.merged_summaries is not None
        assert m.tf.train_writer is not None
        assert m.tf.val_writer is not None

        assert os.path.isdir(m.train_summary_dirpath)
        assert os.path.isdir(m.val_summary_dirpath)

    def test_session_as_argument(self):
        sess = tf.Session()
        m = TestModel(tf_session=sess)
        assert m.tf.session is sess

    def test_session_context_manager_1(self):
        with tf.Session() as sess:
            m = TestModel()
            assert m.tf.session is sess

    def test_session_context_manager_2(self):
        with tf.Session().as_default() as sess:
            m = TestModel()
            assert m.tf.session is sess

    def test_session_context_manager_3(self):
        with tf.Session() as sess:
            m = TestModel(tf_session=sess)
            assert m.tf.session is sess

    def test_session_context_manager_4(self):
        with tf.Session().as_default() as sess:
            m = TestModel(tf_session=sess)
            assert m.tf.session is sess

    def test_save_load(self):
        m = TestModel(model_dirpath='test_model_1')
        m.global_step = 42
        m.save_tf_model()

        tf.reset_default_graph()
        m2 = TestModel(model_dirpath='test_model_1', load_latest=True)
        assert m2.global_step == 42, (m2.global_step)

    def test_save_load_specific_checkpoint(self):
        m = TestModel(model_dirpath='test_model_1')
        for i in xrange(5):
            m.run(m.op)
            m.global_step += 10
            m.save_tf_model()

        for i in xrange(5):
            tf.reset_default_graph()
            global_step = 10 * (i + 1)
            m = TestModel(model_dirpath='test_model_1', global_step=global_step, load_latest=True)
            assert m.run(m.x) == i + 1
            assert m.global_step == global_step

    def test_multiple_models_separate_graphs(self):
        m1 = TestModel(model_dirpath='test_model_1')
        m1.run(m1.x.assign(42))
        m1.save_tf_model()

        tf.reset_default_graph()

        m2 = TestModel(model_dirpath='test_model_2')
        m2.run(m2.x.assign(1337))
        m2.save_tf_model()

        tf.reset_default_graph()

        with tf.Graph().as_default():
            m1 = TestModel(model_dirpath='test_model_1', load_latest=True)
            assert m1.run(m1.x) == 42

        with tf.Graph().as_default():
            m2 = TestModel(model_dirpath='test_model_2', load_latest=True)
            assert m2.run(m2.x) == 1337

    def test_multiple_models_separate_graphs_and_sessions(self):
        m1 = TestModel(model_dirpath='test_model_1')
        m1.run(m1.x.assign(42))
        m1.save_tf_model()

        tf.reset_default_graph()

        m2 = TestModel(model_dirpath='test_model_2')
        m2.run(m2.x.assign(1337))
        m2.save_tf_model()

        tf.reset_default_graph()

        with tf.Graph().as_default():
            with tf.Session().as_default():
                m1 = TestModel(model_dirpath='test_model_1', load_latest=True)
                assert m1.run(m1.x) == 42

        with tf.Graph().as_default():
            with tf.Session().as_default():
                m2 = TestModel(model_dirpath='test_model_2', load_latest=True)
                assert m2.run(m2.x) == 1337

    def test_multiple_models_variable_scopes(self):
        with tf.variable_scope('model1'):
            m1 = TestModel(model_dirpath='test_model_1')
            m1.run(m1.x.assign(42))
            m1.save_tf_model()

        with tf.variable_scope('model2'):
            m2 = TestModel(model_dirpath='test_model_2')
            m2.run(m2.x.assign(1337))
            m2.save_tf_model()

        tf.reset_default_graph()

        with tf.variable_scope('model1'):
            m1 = TestModel(model_dirpath='test_model_1', load_latest=True)
            assert m1.run(m1.x) == 42

        with tf.variable_scope('model2'):
            m2 = TestModel(model_dirpath='test_model_2', load_latest=True)
            assert m2.run(m2.x) == 1337

    def test_partial_save_load_via_list(self):
        m = TestModel(model_dirpath='test_model_1')
        m.tf_latest_saver = tf.train.Saver(var_list=[m.x, m.y])  # save only x, y
        m.global_step = 1111
        m.run(m.x.assign(2222))
        m.run(m.y.assign(3333.))
        m.run(m.z.assign(4444.))
        m.save_tf_model()

        tf.reset_default_graph()

        m = TestModel(model_dirpath='test_model_1')
        # here if corresponding `var_list` is not provided,
        # an error will be raised since saver will expect
        # to find all the variables in the checkpoint, including
        # `global_step` and `z`, which we intentionally haven't saved
        m.tf_latest_saver = tf.train.Saver(var_list=[m.x, m.y])
        TensorFlowModel.load_tf_model(m.model_dirpath,
                                      m.tf.session,
                                      m.tf_latest_saver)

        assert m.global_step == 0
        assert m.run(m.x) == 2222
        assert_almost_equal(3333., m.run(m.y))
        assert_almost_equal(0., m.run(m.z))

    def test_partial_save_load_via_dict(self):
        m = TestModel(model_dirpath='test_model_1')
        m.tf_latest_saver = tf.train.Saver(var_list={'x': m.x, 'y': m.y})  # save only x, y
        m.global_step = 1111
        m.run(m.x.assign(2222))
        m.run(m.y.assign(3333.))
        m.run(m.z.assign(4444.))
        m.save_tf_model()

        tf.reset_default_graph()

        m = TestModel(model_dirpath='test_model_1')
        m.tf_latest_saver = tf.train.Saver(var_list={'x': m.x, 'y': m.y})
        TensorFlowModel.load_tf_model(m.model_dirpath,
                                      m.tf.session,
                                      m.tf_latest_saver)

        assert m.global_step == 0
        assert m.run(m.x) == 2222
        assert_almost_equal(3333., m.run(m.y))
        assert_almost_equal(0., m.run(m.z))

    def test_run_summary(self):
        m = TestModelWithSummary(model_dirpath='test_model_1')

        m.run_summary(train=True)
        assert os.listdir(m.train_summary_dirpath)
        assert not os.listdir(m.val_summary_dirpath)

        m.run_summary(train=False)
        assert os.listdir(m.val_summary_dirpath)

    def tearDown(self):
        tf.reset_default_graph()
        self.cleanup()
