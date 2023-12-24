# Running experiments
The `experiments/defaults.yml` file contains the tuned hyperparameters and does not need to be changed.

The actual experiments can be run using `python main.py`. For each Figure, here are the commands that need to be changed from the default in the provided `main.py` file.

For Figure 4 (Main results):

```sh
python main.py --obs_coeff=6 --intensity_cov_only=False --num_patients=200 --importance_weighting=True --multitask=False
```

You can run this for the different models:
```text
TE-CDE: --importance_weighting=False –multitask=False
TESAR-CDE (Two step): --importance_weighting=True –multitask=False
TESAR-CDE (Multitask): --importance_weighting=True –multitask=True
```

Changing `obs_coeff` in `{0, 2, …, 10}` will get you the results for Figure 4.
 

For Figure 5 (Observation scarcity):

```sh
python main.py --obs_coeff=4 --intensity_cov_only=False --num_patients=200 --importance_weighting=True --multitask=False --max_intensity=1
```

Here, the sampling scarcity reported in the paper is `1/max_intensity`. So you will have to change `max_intensity` in `{1, 1/2, 1/3, 1/4}`.

Again, you can change the different models as before to get the different experiments.

For Figure 6 (Outcome-unrelated sampling):

```sh
python main.py --obs_coeff=4 --intensity_cov_only=True --intensity_cov=10 --num_patients=200 --importance_weighting=True --multitask=False --max_intensity=1
```

Again, you can change the different models and `obs_coeff` as before.
