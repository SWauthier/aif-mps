# Planning with tensor networks based on active inference

AI planning using matrix product states to model sequences and the action selection algorithm provided by active inference.
A model-based approach to learning and planning i.e. the agent’s generative model is a tensor network.
Uses an update scheme which alternates between single-site and two-site updates, and allows dynamic variation of bond dimensions, while avoiding exploding bond dimensions as well as speeding up computation.

## Usage
### Running experiments
Two experiment files can be found inside ```mpstwo```: ```frozenlake.py``` and ```tmaze.py```.<br/>
E.g. to run frozenlake experiments with default hyperparameters:
```
python main.py frozenlake run
```
Hyperparameters can be adjusted through the command line:
```
python main.py frozenlake run --log_dir experiment1 --environment.is_slippery False
```
<br/><br/>
Alternatively, run experiments through:
```
python mpstwo/frozenlake.py run
```
Or with hyperparameters:
```
python mpstwo/frozenlake.py run model.init_mode="positive" model.cutoff=0.1 pool.rollouts=204
```
### Hyperparameters
The hyperparameters which can be adjusted can be found in the ```config``` variable near the top of each file.
