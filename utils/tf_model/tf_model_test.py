import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from shutil import rmtree

import tensorflow as tf

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
    def _make_tf_model(self):
        super(TestModelWithSummary, self)._make_tf_model()
        tf.summary.scalar(self.x.name, self.x)


class TestTensorFlowModel(tf.test.TestCase):
    def __init__(self, *args, **kwargs):
        self.model_dirpaths = ['test_model_{}'.format(i) for i in xrange(2)]
        super(TestTensorFlowModel, self).__init__(*args, **kwargs)

    def test_vars_and_ops(self):
        m = TestModel()

        self.assertEqual(m.global_step, 0)
        self.assertEqual(m.run(m.tf.global_step), 0)
        self.assertEqual(m.run(m.x), 0)
        self.assertAlmostEqual(m.run(m.y), 0.)
        self.assertAlmostEqual(m.run(m.z), 0.)

        m.run(m.op)

        self.assertEqual(m.run(m.x), 1)

    def test_global_step(self):
        with self.test_session() as sess:
            global_step = tf.train.get_or_create_global_step(sess.graph)
            m = TestModel()
            self.assertIs(m.tf.global_step, global_step)

            m.global_step = 1
            self.assertEqual(m.global_step, 1)
            m.global_step = 0
            self.assertEqual(m.global_step, 0)
            m.global_step += 10
            self.assertEqual(m.global_step, 10)

    def test_summaries_not_created(self):
        m = TestModel()

        self.assertIsNone(m.tf.merged_summaries)
        self.assertIsNone(m.tf.train_writer)
        self.assertIsNone(m.tf.val_writer)

        self.assertFalse(os.path.isdir(m.train_summary_dirpath))
        self.assertFalse(os.path.isdir(m.val_summary_dirpath))

    def test_summaries_created(self):
        m = TestModelWithSummary(model_dirpath=self.model_dirpaths[0])

        self.assertIsNotNone(m.train_summary_dirpath)
        self.assertIsNotNone(m.val_summary_dirpath)
        self.assertIsNotNone(m.tf.merged_summaries)
        self.assertIsNone(m.tf.train_writer)
        self.assertIsNone(m.tf.val_writer)

        m.run_summary(train=True)

        self.assertIsNotNone(m.tf.train_writer)
        self.assertIsNone(m.tf.val_writer)

        m.run_summary(train=False)

        self.assertIsNotNone(m.tf.train_writer)
        self.assertIsNotNone(m.tf.val_writer)

    def test_session_as_argument(self):
        sess = tf.Session()
        m = TestModel(tf_session=sess)
        self.assertIs(m.tf.session, sess)

    def test_session_context_manager_1(self):
        with tf.Session() as sess:
            m = TestModel()
            self.assertIs(m.tf.session, sess)

    def test_session_context_manager_2(self):
        with tf.Session().as_default() as sess:
            m = TestModel()
            self.assertIs(m.tf.session, sess)

    def test_session_context_manager_3(self):
        with tf.Session() as sess:
            m = TestModel(tf_session=sess)
            self.assertIs(m.tf.session, sess)

    def test_session_context_manager_4(self):
        with tf.Session().as_default() as sess:
            m = TestModel(tf_session=sess)
            self.assertIs(m.tf.session, sess)

    def test_save_load(self):
        m = TestModel(model_dirpath=self.model_dirpaths[0])
        m.global_step = 42
        m.save_tf_model()

        tf.reset_default_graph()
        m2 = TestModel(model_dirpath=self.model_dirpaths[0], load_latest=True)
        self.assertEqual(m2.global_step, 42)

    def test_save_load_specific_checkpoint(self):
        m = TestModel(model_dirpath=self.model_dirpaths[0])
        for i in xrange(5):
            m.run(m.op)
            m.global_step += 10
            m.save_tf_model()

        for i in xrange(5):
            tf.reset_default_graph()
            global_step = 10 * (i + 1)
            m = TestModel(model_dirpath=self.model_dirpaths[0], global_step=global_step, load_latest=True)

            self.assertEqual(m.run(m.x), i + 1)
            self.assertEqual(m.global_step, global_step)

    def test_multiple_models_separate_graphs(self):
        m1 = TestModel(model_dirpath=self.model_dirpaths[0])
        m1.run(m1.x.assign(42))
        m1.save_tf_model()

        tf.reset_default_graph()

        m2 = TestModel(model_dirpath=self.model_dirpaths[1])
        m2.run(m2.x.assign(1337))
        m2.save_tf_model()

        tf.reset_default_graph()

        with tf.Graph().as_default():
            m1 = TestModel(model_dirpath=self.model_dirpaths[0], load_latest=True)
            self.assertEqual(m1.run(m1.x), 42)

        with tf.Graph().as_default():
            m2 = TestModel(model_dirpath=self.model_dirpaths[1], load_latest=True)
            self.assertEqual(m2.run(m2.x), 1337)

    def test_multiple_models_separate_graphs_and_sessions(self):
        m1 = TestModel(model_dirpath=self.model_dirpaths[0])
        m1.run(m1.x.assign(42))
        m1.save_tf_model()

        tf.reset_default_graph()

        m2 = TestModel(model_dirpath=self.model_dirpaths[1])
        m2.run(m2.x.assign(1337))
        m2.save_tf_model()

        tf.reset_default_graph()

        with tf.Graph().as_default():
            with tf.Session().as_default():
                m1 = TestModel(model_dirpath=self.model_dirpaths[0], load_latest=True)
                self.assertEqual(m1.run(m1.x), 42)

        with tf.Graph().as_default():
            with tf.Session().as_default():
                m2 = TestModel(model_dirpath=self.model_dirpaths[1], load_latest=True)
                self.assertEqual(m2.run(m2.x), 1337)

    def test_multiple_models_variable_scopes(self):
        with tf.variable_scope('model1'):
            m1 = TestModel(model_dirpath=self.model_dirpaths[0])
            m1.run(m1.x.assign(42))
            m1.save_tf_model()

        with tf.variable_scope('model2'):
            m2 = TestModel(model_dirpath=self.model_dirpaths[1])
            m2.run(m2.x.assign(1337))
            m2.save_tf_model()

        tf.reset_default_graph()

        with tf.variable_scope('model1'):
            m1 = TestModel(model_dirpath=self.model_dirpaths[0], load_latest=True)
            self.assertEqual(m1.run(m1.x), 42)

        with tf.variable_scope('model2'):
            m2 = TestModel(model_dirpath=self.model_dirpaths[1], load_latest=True)
            self.assertEqual(m2.run(m2.x), 1337)

    def test_partial_save_load_via_list(self):
        m = TestModel(model_dirpath=self.model_dirpaths[0])
        m.tf_latest_saver = tf.train.Saver(var_list=[m.x, m.y])  # save only x, y
        m.global_step = 1111
        m.run(m.x.assign(2222))
        m.run(m.y.assign(3333.))
        m.run(m.z.assign(4444.))
        m.save_tf_model()

        tf.reset_default_graph()

        m = TestModel(model_dirpath=self.model_dirpaths[0])
        # here if corresponding `var_list` is not provided,
        # an error will be raised since saver will expect
        # to find all the variables in the checkpoint, including
        # `global_step` and `z`, which we intentionally haven't saved
        m.tf_latest_saver = tf.train.Saver(var_list=[m.x, m.y])
        TensorFlowModel.load_tf_model(m.model_dirpath,
                                      m.tf.session,
                                      m.tf_latest_saver)
        self.assertEqual(m.global_step, 0)
        self.assertEqual(m.run(m.x), 2222)
        self.assertAlmostEqual(m.run(m.y), 3333.)
        self.assertAlmostEqual(m.run(m.z), 0.)

    def test_partial_save_load_via_dict(self):
        m = TestModel(model_dirpath=self.model_dirpaths[0])
        m.tf_latest_saver = tf.train.Saver(var_list={'x': m.x, 'y': m.y})  # save only x, y
        m.global_step = 1111
        m.run(m.x.assign(2222))
        m.run(m.y.assign(3333.))
        m.run(m.z.assign(4444.))
        m.save_tf_model()

        tf.reset_default_graph()

        m = TestModel(model_dirpath=self.model_dirpaths[0])
        m.tf_latest_saver = tf.train.Saver(var_list={'x': m.x, 'y': m.y})
        TensorFlowModel.load_tf_model(m.model_dirpath,
                                      m.tf.session,
                                      m.tf_latest_saver)
        self.assertEqual(m.global_step, 0)
        self.assertEqual(m.run(m.x), 2222)
        self.assertAlmostEqual(m.run(m.y), 3333.)
        self.assertAlmostEqual(m.run(m.z), 0.)

    def test_run_summary(self):
        m = TestModelWithSummary(model_dirpath=self.model_dirpaths[0])

        self.assertFalse(os.path.isdir(m.train_summary_dirpath))
        self.assertFalse(os.path.isdir(m.val_summary_dirpath))

        m.run_summary(train=True)

        self.assertTrue(os.listdir(m.train_summary_dirpath))
        self.assertFalse(os.path.isdir(m.val_summary_dirpath))

        m.run_summary(train=False)

        self.assertTrue(os.listdir(m.train_summary_dirpath))
        self.assertTrue(os.listdir(m.val_summary_dirpath))

    def tearDown(self):
        tf.reset_default_graph()

        for d in self.model_dirpaths:
            try:
                rmtree(d)
            except OSError:
                pass


if __name__ == '__main__':
    tf.test.main()
