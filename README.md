# Neural Inverse Rendering for General Reflectance Photometric Stereo (ICML 2018)

We provide an implementation of the photometric stereo method presented in the following paper. If you find our code useful for your research, please cite our paper.
```
@inproceedings{Taniai18,
  author    = {Tatsunori Taniai and
               Takanori Maehara},
  title     = {{Neural Inverse Rendering for General Reflectance Photometric Stereo}},
  booktitle = {{Proceedings of the 35th International Conference on Machine Learning (ICML)}},
  pages     = {4864--4873},
  year      = {2018},
}
```
Links [[Paper]](http://proceedings.mlr.press/v80/taniai18a.html)  [[Project]](https://taniai.space/projects/tm18_neuralps/)

## Running Environments
- Python 3.5+ 
- Chainer 3.4 (or maybe higher version is acceptable)
- numpy
- CUDA and cuDNN (if use the GPU mode "-g 0")

## How to Run?
Move to chainer_code directory and enter the following command.

For CPU mode
```
python train.py -t 0
```

For GPU dode
```
python train.py -t 0 -g 0
```

## Options
| Options | Type | Settings |
----|---- |---
| --gpu (-g) | int | GPU ID to use |
| --target (-t) | int (0-9) | Scene number from 0 to 9 (ball to reading) |
