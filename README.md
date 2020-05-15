# mean-shift-rejection
## mxnet implementation
`train_cifar100.py`
 * This is the entry point. Currently hard-coded for the cifar100 dataset (will automatically download to `~/.mxnet/datasets`), but should be swap out
 * Make note of the various cmd line args
 * I implemented `zmg_norm()`, which should apply ZMG to the gradients when `--zmg` > 0, but I have yet to validate that it works.

`environment.yml`
* The conda environment I used to run this code

`.gitignore`
* This tells git what files to ignore from version control. (I highly suggest getting familiar with git, it's indispensable)

`params`
* Saved models during training get saved to this folder by default

`models`
* Implementations copied from official gluon library (only imports altered to work here)
* Note that `models/__init__.py` contains `get_model()`, which contains a dict mapping model names to implementations
* `models/resnetv1b.py` contains all official mxnet resnet variants aside from original resnet implementation