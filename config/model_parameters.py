RESNET50_MODEL_PARAMS = [
    {"filters": [64, 64, 256],
     "kernel_size": [(1, 1), (3, 3), (1, 1)],
     "strides": [[(1, 1), (1, 1)], (1, 1), (1, 1)],
     "padding": ['valid', 'same', 'valid'],
     "num_blocks": 3
     },
    {"filters": [128, 128, 512],
     "kernel_size": [(1, 1), (3, 3), (1, 1)],
     "strides": [[(2, 2), (1, 1)], (1, 1), (1, 1)],
     "padding": ['valid', 'same', 'valid'],
     "num_blocks": 4
     },
    {"filters": [256, 256, 1024],
     "kernel_size": [(1, 1), (3, 3), (1, 1)],
     "strides": [[(2, 2), (1, 1)], (1, 1), (1, 1)],
     "padding": ['valid', 'same', 'valid'],
     "num_blocks": 6
     },
    {"filters": [512, 512, 2048],
     "kernel_size": [(1, 1), (3, 3), (1, 1)],
     "strides": [[(2, 2), (1, 1)], (1, 1), (1, 1)],
     "padding": ['valid', 'same', 'valid'],
     "num_blocks": 3
     }
]
