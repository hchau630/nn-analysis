[
    {
        "model_names": ["identity"],
        "layers": [None],
        "metrics": [
            ["decode", 0],
            ["dimensionality", 0],
            ["fact", 0],
            ["generalize", 0],
            ["neural_fits", 0],
            ["rdm", 0],
            ["sparsity", 0],
            ["curve", 0],
            ["curve", 1],
            ["trajectory", 0],
        ],
    },
    {
        "model_names": [
            "moco_control",
            "moco_CD",
            "moco_CF",
            "moco_CDF",
            "barlow_control",
            "barlow_CD",
            "barlow_CF",
            "barlow_CDF",
            "barlow_P",
            "barlow_PF",
        ],
        "epochs": [49,50],
        "layers": [None], # All layers
        "metrics": [
            ["decode", 0],
            ["dimensionality", 0],
            ["fact", 0],
            ["generalize", 0],
            ["neural_fits", 0],
            ["rdm", 0],
            ["sparsity", 0],
            ["curve", 0],
            ["curve", 1],
            ["trajectory", 0],
        ]
    },
    {
        "model_names": [
            "barlow_P",
            "barlow_P_projector",
        ],
        "epochs": [82,83], # epoch 82
        "layers": [None], # All layers
        "metrics": [
            # ["decode", 0],
            # ["dimensionality", 0],
            # ["fact", 0],
            # ["generalize", 0],
            # ["neural_fits", 0],
            # ["rdm", 0],
            # ["sparsity", 0],
            ["curve", 0],
            ["curve", 1],
            # ["trajectory", 0],
        ]
    },
    {
        "model_names": [
            "barlow_control",
            "barlow_control_projector",
        ],
        "epochs": [54,55], # epoch 82
        "layers": [None], # All layers
        "metrics": [
            # ["decode", 0],
            # ["dimensionality", 0],
            # ["fact", 0],
            # ["generalize", 0],
            # ["neural_fits", 0],
            # ["rdm", 0],
            # ["sparsity", 0],
            ["curve", 0],
            ["curve", 1],
            # ["trajectory", 0],
        ]
    },
    # {
    #     "model_names": [
    #         "moco_control",
    #         "moco_CD",
    #         "moco_CF",
    #         "moco_CDF",
    #         "barlow_control",
    #         "barlow_CD",
    #         "barlow_CF",
    #         "barlow_CDF",
    #         "barlow_P",
    #         "barlow_PF",
    #     ],
    #     "epochs": [0,50],
    #     "layers": [15,None], # 15, 16 layers
    #     "metrics": [
    #         ["decode", 0],
    #         ["dimensionality", 0],
    #         ["fact", 0],
    #         ["generalize", 0],
    #         ["neural_fits", 0],
    #         ["rdm", 0],
    #         ["sparsity", 0],
    #     ]
    # }
]