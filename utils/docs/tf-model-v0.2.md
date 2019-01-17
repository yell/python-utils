## TensorFlowModel

TODO: latest simplifications!

### Typical usage
```python
class MyModel(TensorFlowModel):
    def __init__(self, ..., *args, **kwargs):
        # define attributes
        super(MyModel, self).__init__(*args, **kwargs)  # ->
        # -> builds all infrastructure and calls `_make_tf_model`

    def _make_tf_model():
        # build you TF model here

    def partial_fit(X, ...):
        self.run_in_tf_session(self._train_op, {self.x_ph: X})
```

### TF Session
Similar to how `Keras`, `edward` do it:
* can be passed via `tf_session`:
    ```python
    m = MyModel(..., tf_session=tf.Session(), ...)
    # or
    m = MyModel(..., tf_session=tf.InteractiveSession(), ...)
    # then can m.op.eval()
    ```
* otherwise get globally define one:
    ```python
    with tf.Session() as sess:
        m = MyModel()  # will use `sess`
    ```
    OR
    ```python
    sess = tf.InteractiveSession()
    m = MyModel()  # will use interactive session
    ```
* otherwise creates one
* `get_session()`
* `run_in_tf_session(...)`

### TF Graph
* can be passed via `tf_graph`
* otherwise is taken from session (`graph = sess.graph`)
* is passed to `train` writer for visualization

### Global step
* again can be passed via `global_step`
* otherwise get globally defined or create if not exist (`tf.train.get_or_create_global_step( graph )`)
* by default used in summaries and custom summaries (though any value of `global_step` can be provided)
* can be used by optimizers etc.
* property getter/setter: `model.global_step -> int`, `model.global_step <- int`
* `model.inc_global_step()`

### TF Saver
* is always created, as there is at least 1 variable -- `global_step` [no error because of no variables possible]
* // though no writing on disk till calling `save_tf_model`
* control via `tf_saver_kwargs`, `tf_saver_kwargs_fn( model )` // TODO: simplify?

#### Saving
* `save_tf_model()`
* by default does not ``
* additionally can pass `*tf_save_args, **tf_save_kwargs` for `tf.Saver.save(...)` in `save_tf_model(...)`

#### Restoring
* `TensorFlowModel.load_tf_model(dirpath, tf_session, tf_saver=None, global_step=None, *tf_restore_args, **tf_restore_kwargs)`
* if `global_step` is not provided, loads last checkpoint in `dirpath`, otherwise load checkpoint with given `global_step` (int)
* if `tf_saver` is not passed, loads via `tf.train.import_meta_graph`

* loads the model from disk in the constructor if `load=True`
* to load specific model in constructor, simply pass in `global_step` (naturally loaded model will have the same `global_step`)
* if model loads in the constructor, taking into acc. that `_make_tf_model` is also called in the ctor (before loading), there is some flexibility compared by loading model by `tf.train.import_meta_graph` (though this is still can be done within this class), that is e.g. **partial save/load**:
    ```python
    m = MyModel(..., [load=True, ]
                tf_saver_kwargs_fn=lambda m: {'var_list': [m.var_to_keep1,
                                                           m.var_to_keep2]})
    ```
    all other uninitialized variables then initialized. Then allow to add or remove variables from the model while still being able to load (remaining) weights from the checkpoints
* load multiple models in 1 script:
    1. using separate graphs
        ```python
        with tf.Graph().as_default():
            m1 = MyModel(..., load=True, ...)

        with tf.Graph().as_default():
            m2 = MyModel(..., load=True, ...)
        ```

    2. using separate graphs and sessions
        ```python
        with tf.Graph().as_default():
            with tf.Session().as_default() as sess:
                m1 = MyModel(..., load=True, ...)
                m1.get_session() is sess  # -> True

        with tf.Graph().as_default():
            with tf.Session().as_default():
                m2 = MyModel(..., load=True, ...)
        ```

    3. using same graph but with variable scopes
        ```python
        with tf.variable_scope('model1'):
            m1 = MyModel(...)
            m1.save_tf_model()

        with tf.variable_scope('model2'):
            m2 = MyModel(...)
            m2.save_tf_model()

        # later
        with tf.variable_scope('model1'):
            m1 = MyModel(... load=True, ...)

        with tf.variable_scope('model2'):
            m2 = MyModel(..., load=True, ...)
        ```

    4. using variable scopes but for already saved models
        ```python
        # add prefix of the variable scope (e.g. 'model1')
        # for variable names in the saved checkpoint using
        # https://gist.github.com/batzner/7c24802dd9c5e15870b4b56e22135c96
        # or
        # https://stackoverflow.com/a/41829414/7322931
        # (neither is tested though)

        # after that
        with tf.variable_scope('model1'):
            m1 = MyModel(... load=True, ...)

### Summaries
* `train` and `val` writer (cuz you want both!)
* writers created (+ corresp. folders) **only** if there is at least 1 summary in `_make_tf_model`
  OR if Custom summaries are called
* `run_summary(summary_ops=None)` run `summary_ops` in corresponding session, by default it is `tf.merged_summaries()`
* if they are expensive to (re)compute, they can be, of course, ran in `run_in_tf_session([train_op, expensive_summaries])`

### Custom summaries
use tensorboard w/o TF operations:
* `log_scalars( numpy )`
* `log_images( numpy )`
* `log_histogram( numpy )`

### Final remarks
* `_make_tf_model` can be implemented in pretty much arbitrary way, it can be designed to build the TF graph lazily, e.g. (https://danijar.com/structuring-your-tensorflow-models/) [tricks]
