### Preparation
To setup conda environment with all required dependencies run:

```bash
conda env create -f rl-env.yaml
```

If you want to use venv, create new virtual environment with python 3.11, activate it and run:

```bash
pip install -r requirements.txt
```

### Setup your experiment
You can modify student's RL algorithm configuration by changing adequate fields in the `rl_config.json`.

### Run experiment
From home directory run:

```bash
python run_experiment.py --scenario [env_name] --teacher_type [teacher_name]
```

Possible environment names:
- bipedal_walker

Possible teacher names:
- alpgmm
- oracle
- random
- rl

### Test the final model

Run

```bash
python test_final_model.py --log-dir [path/to/logs] --steps [number of steps, ie. 2000] --param stump_height=[number in range (0, 1)] --param stump_distance=[number in range (0, 1)]
```