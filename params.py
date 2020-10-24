from libraries import *

hyperparameter_space = [space.Integer(3, 10, name = 'depth'),
                        space.Real(0.005, 0.01, name = 'learning_rate'),
                        space.Real(0.1, 0.9, name = 'bagging_temperature'),
                        space.Integer(1, 10, name = 'l2_leaf_reg'),
                        space.Integer(1, 255, name = 'border_count')
                        ]

reg_hyperparameter_space = [space.Integer(3, 10, name = 'depth'),
                        space.Real(0.005, 0.01, name = 'learning_rate'),
                        space.Real(0.1, 0.9, name = 'bagging_temperature')
                        ]


