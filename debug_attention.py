#!/usr/bin/env python3
"""
Debug script to understand why attention mechanism isn't being created
"""

from azchess.config import Config
from azchess.model.resnet import NetConfig, ChessAttention
from azchess.model import PolicyValueNet
import torch

def debug_attention():
    cfg = Config.load('config.yaml')
    model_cfg = cfg.model()

    print('=== CONFIG VALUES ===')
    print(f'attention: {model_cfg.get("attention")}')
    print(f'attention_heads: {model_cfg.get("attention_heads")}')
    print(f'attention_every_k: {model_cfg.get("attention_every_k")}')
    print(f'blocks: {model_cfg.get("blocks")}')

    # Create NetConfig
    net_config = NetConfig(**model_cfg)
    print(f'\n=== NETCONFIG VALUES ===')
    print(f'attention: {net_config.attention}')
    print(f'attention_heads: {net_config.attention_heads}')
    print(f'attention_every_k: {net_config.attention_every_k}')
    print(f'blocks: {net_config.blocks}')

    # Simulate the tower building logic
    print(f'\n=== SIMULATING TOWER BUILDING ===')
    C = net_config.channels
    blocks = net_config.blocks
    attention = net_config.attention
    att_every = net_config.attention_every_k

    print(f'Channels: {C}')
    print(f'Blocks: {blocks}')
    print(f'Attention enabled: {attention}')
    print(f'Attention every: {att_every}')

    tower_layers = []
    attention_layers_added = []

    for i in range(blocks):
        print(f'Block {i}: Adding ResidualBlock')
        # Simulate adding residual block (we won't create it, just log)

        # Check attention condition
        if attention and att_every > 0 and (i % att_every) == (att_every - 1):
            print(f'Block {i}: ADDING ChessAttention after this block!')
            attention_layers_added.append(i)
            # Simulate adding attention (we won't create it, just log)
        else:
            print(f'Block {i}: No attention added (condition not met)')

    print(f'\n=== RESULTS ===')
    print(f'Attention layers would be added after blocks: {attention_layers_added}')
    print(f'Total attention layers: {len(attention_layers_added)}')

    # Now actually create the model and check
    print(f'\n=== ACTUAL MODEL CREATION ===')
    model = PolicyValueNet(net_config)

    actual_attention_count = 0
    actual_attention_params = 0

    for name, module in model.named_modules():
        if 'attention' in name.lower() or 'attn' in name.lower():
            actual_attention_count += 1
            actual_attention_params += sum(p.numel() for p in module.parameters())
            print(f'Found attention module: {name}')

    print(f'Actual attention layers: {actual_attention_count}')
    print(f'Actual attention parameters: {actual_attention_params}')

    # Check tower structure
    print(f'\n=== TOWER STRUCTURE ===')
    if hasattr(model, 'tower'):
        print(f'Tower has {len(model.tower)} layers:')
        for i, layer in enumerate(model.tower):
            layer_type = type(layer).__name__
            if 'Attention' in layer_type:
                print(f'  Layer {i}: {layer_type} - ATTENTION!')
            else:
                print(f'  Layer {i}: {layer_type}')

if __name__ == '__main__':
    debug_attention()
