# python-utils
This repository contains:
* [`TensorFlowModel`](utils/tf_model/tf_model.py) (class encapsulating basic TF infrastructure for executing computation, saving/restoring, and visualization)
* [TensorFlow utils](utils/tf_utils/)
* [PyTorch utils](utils/torch_utils/)
* [other python utils](utils/)

## How to install
```bash
git clone https://github.com/monsta-hd/utils.git
cd utils
pip install -r requirements.txt
```
After installation, tests can be run with:
```bash
make test
```
