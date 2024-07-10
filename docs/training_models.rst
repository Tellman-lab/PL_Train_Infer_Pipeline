Training a Model Tutorial
=====

Train a single model
-------

Run default model.

```python ./st_water_seg/fit.py```


Customize model training.

1. Inspect ```./st_water_seg/conf/config.yaml```
2. Overwrite hyperparameter

```python ./st_water_seg/fit.py lr=1e-5```


Train a cross-validation experiment
-------

