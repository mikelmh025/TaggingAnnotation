# Colab version https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Organizing_Hyperparameter_Sweeps_in_PyTorch_with_W%26B.ipynb#scrollTo=hyHDTrZAUZH7


import wandb

wandb.login()

# Sweep config
sweep_config = {
    'method': 'random'
    }

# Metric to optimize
metric = {
    'name': 'loss',
    'goal': 'minimize'   
    }
sweep_config['metric'] = metric

# Parameters to optimize
parameters_dict = {
    'optimizer': {
        'values': ['adam', 'sgd']
        },
    'fc_layer_size': {
        'values': [128, 256, 512]
        },
    'dropout': {
          'values': [0.3, 0.4, 0.5]
        },
    }
sweep_config['parameters'] = parameters_dict


# It's often the case that there are hyperparameters that we don't 
# want to vary in this Sweep, but which we still want to set in our sweep_config.
# 
# In that case, we just set the value directly:
parameters_dict.update({
    'epochs': {
        'value': 1}
    })

# For a grid search, that's all you ever need.

# For a random search, all the values of a parameter are equally likely to be chosen on a given run.

# If that just won't do, you can instead specify a named distribution, plus its parameters, 
# like the mean mu and standard deviation sigma of a normal distribution.

# See more on how to set the distributions of your random variables here.
parameters_dict.update({
    'learning_rate': {
        # a flat distribution between 0 and 0.1
        'distribution': 'uniform',
        'min': 0,
        'max': 0.1
      },
    'batch_size': {
        # integers between 32 and 256
        # with evenly-distributed logarithms 
        'distribution': 'q_log_uniform_values',
        'q': 8,
        'min': 32,
        'max': 256,
      }
    })

import pprint

pprint.pprint(sweep_config)

sweep_id = wandb.sweep(sweep_config, project="pytorch-intro")
a=1