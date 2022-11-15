SegformerConfig = {
    'nano': {
        'type': 'Segformer',
        'pretrained': 'none',
        'backbone': {
            'type': 'MixVisionTransformer',
            'in_channels': 3,
            'embed_dims': 32,
            'num_stages': 4,
            'num_layers': [2, 2, 2, 2],
            'num_heads': [1, 2, 5, 8],
            'patch_sizes': [7, 3, 3, 3],
            'sr_ratios': [8, 4, 2, 1],
            'out_indices': (0, 1, 2, 3),
            'mlp_ratio': 4,
            'qkv_bias': True,
            'drop_rate': 0.0,
            'attn_drop_rate': 0.0,
            'drop_path_rate': 0.1
        },
        'decode_head': {
            'type': 'SegformerHead',
            'in_channels': [32, 64, 160, 256],
            'channels': 256,
            'dropout_ratio': 0.1,
            'num_classes': -1,
            'norm_cfg': {'type': 'BN'},
            'loss_decode': {'type': 'CrossEntropyLoss', 'use_sigmoid': False, 'loss_weight': 1.0}
        }
    },
    'm': {
        'type': 'Segformer',
        'pretrained': 'none',
        'backbone': {
            'type': 'MixVisionTransformer',
            'in_channels': 3,
            'embed_dims': 64,
            'num_stages': 4,
            'num_layers': [3, 4, 6, 3],
            'num_heads': [1, 2, 5, 8],
            'patch_sizes': [7, 3, 3, 3],
            'sr_ratios': [8, 4, 2, 1],
            'out_indices': (0, 1, 2, 3),
            'mlp_ratio': 4,
            'qkv_bias': True,
            'drop_rate': 0.0,
            'attn_drop_rate': 0.0,
            'drop_path_rate': 0.1
        },
        'decode_head': {
            'type': 'SegformerHead',
            'in_channels': [64, 128, 320, 512],
            'in_index': [0, 1, 2, 3],
            'channels': 256,
            'dropout_ratio': 0.1,
            'num_classes': -1,
            'norm_cfg': {'type': 'BN'},
            'align_corners': False,
            'loss_decode': {'type': 'CrossEntropyLoss', 'use_sigmoid': False, 'loss_weight': 1.0}
        }
    },
    'xl': {
        'type': 'Segformer',
        'pretrained': 'none',
        'backbone': {
            'type': 'MixVisionTransformer',
            'in_channels': 3,
            'embed_dims': 64,
            'num_stages': 4,
            'num_layers': [3, 6, 40, 3],
            'num_heads': [1, 2, 5, 8],
            'patch_sizes': [7, 3, 3, 3],
            'sr_ratios': [8, 4, 2, 1],
            'out_indices': (0, 1, 2, 3),
            'mlp_ratio': 4,
            'qkv_bias': True,
            'drop_rate': 0.0,
            'attn_drop_rate': 0.0,
            'drop_path_rate': 0.1
        },
        'decode_head': {
            'type': 'SegformerHead',
            'in_channels': [64, 128, 320, 512],
            'in_index': [0, 1, 2, 3],
            'channels': 256,
            'dropout_ratio': 0.1,
            'num_classes': -1,
            'norm_cfg': {'type': 'BN'},
            'align_corners': False,
            'loss_decode': {'type': 'CrossEntropyLoss', 'use_sigmoid': False, 'loss_weight': 1.0}
        }
    }
}
