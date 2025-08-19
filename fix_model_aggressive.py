#!/usr/bin/env python3

"""
Aggressive Model Fix Script - Completely reinitialize policy head to resolve uniform outputs
"""

import torch
import torch.nn as nn
import chess
import numpy as np
from azchess.model.resnet import PolicyValueNet
from azchess.config import Config
from azchess.encoding import encode_board
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import logging

def fix_model_aggressive():
    print("🔧 Aggressively fixing model with uniform policy issue...")
    
    # Load config and create model
    cfg = Config.load('config.yaml')
    model = PolicyValueNet.from_config(cfg.model())
    
    # Load current checkpoint
    checkpoint_path = 'checkpoints/best.pt'
    state = torch.load(checkpoint_path, map_location='cpu')
    model_state_dict = state.get("model_ema", state.get("model", state))
    model.load_state_dict(model_state_dict)
    
    print("✅ Model loaded from checkpoint")
    
    # Test current policy (should be uniform)
    model.eval()
    board = chess.Board()
    x = torch.from_numpy(encode_board(board)).unsqueeze(0)
    
    with torch.no_grad():
        p, v = model(x)
        p_probs = torch.softmax(p, dim=1)
        
    print(f"Current policy entropy: {-(p_probs * torch.log(p_probs + 1e-8)).sum().item():.4f}")
    print(f"Policy should be diverse, not uniform!")
    
    # Completely reinitialize policy head with much larger weights
    print("\n🔄 Completely reinitializing policy head...")
    
    # Freeze all layers except policy head
    for name, param in model.named_parameters():
        if 'policy' not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
            print(f"Training: {name}")
    
    # Completely reinitialize with much larger weights
    with torch.no_grad():
        # Policy head conv layers - use larger initialization
        nn.init.kaiming_normal_(model.policy_head[0].weight, mode='fan_out', nonlinearity='relu')
        model.policy_head[0].weight.data *= 3.0  # Much larger weights
        
        # Policy FC layer - use much larger initialization
        nn.init.xavier_uniform_(model.policy_fc.weight, gain=3.0)  # Much larger gain
        nn.init.constant_(model.policy_fc.bias, -0.5)  # Larger negative bias
        
        # Scale up existing weights
        model.policy_fc.weight.data *= 5.0  # 5x larger
    
    print("✅ Policy head reinitialized with larger weights")
    
    # Create synthetic training data with strong diversity signals
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=0.005)  # Higher learning rate
    scheduler = CosineAnnealingLR(optimizer, T_max=200)  # More epochs
    
    # Training loop
    model.train()
    for epoch in range(200):  # More training epochs
        optimizer.zero_grad()
        
        # Create a batch of positions
        positions = []
        targets = []
        
        for i in range(16):  # Larger batch
            # Create different board positions
            board = chess.Board()
            if i > 0:
                # Make some random moves to create variety
                for _ in range(min(i, 8)):
                    legal_moves = list(board.legal_moves)
                    if legal_moves:
                        board.push(np.random.choice(legal_moves))
            
            x = torch.from_numpy(encode_board(board)).unsqueeze(0)
            positions.append(x)
            
            # Create target policy with much stronger diversity signals
            target = torch.zeros(1, 4672)
            legal_moves = list(board.legal_moves)
            
            if legal_moves:
                # Much stronger scoring for diversity
                for move in legal_moves:
                    idx = move_to_index(board, move)
                    if idx < len(target[0]):
                        score = 0.1
                        
                        # Much stronger bonuses
                        if move.to_square in [chess.E4, chess.E5, chess.D4, chess.D5]:
                            score += 2.0  # Much stronger center bonus
                        
                        if board.piece_at(move.from_square) and board.piece_at(move.from_square).piece_type == chess.PAWN:
                            score += 1.5  # Stronger pawn bonus
                        
                        if board.piece_at(move.from_square) and board.piece_at(move.from_square).piece_type in [chess.KNIGHT, chess.BISHOP]:
                            score += 1.0  # Stronger development bonus
                        
                        # Add some randomness to break symmetry
                        score += np.random.normal(0, 0.3)
                        
                        target[0, idx] = score
                
                # Normalize to probabilities
                target = torch.softmax(target, dim=1)
            
            targets.append(target)
        
        # Forward pass
        batch_x = torch.cat(positions, dim=0)
        p, v = model(batch_x)
        
        # Policy loss with strong diversity encouragement
        p_probs = torch.softmax(p, dim=1)
        
        # KL divergence loss
        target_probs = torch.cat(targets, dim=0)
        kl_loss = torch.nn.functional.kl_div(
            torch.log(p_probs + 1e-8), 
            target_probs, 
            reduction='batchmean'
        )
        
        # Much stronger entropy regularization
        policy_entropy = -(p_probs * torch.log(p_probs + 1e-8)).sum(dim=1).mean()
        max_entropy = torch.log(torch.tensor(p_probs.shape[1], dtype=torch.float32))
        entropy_penalty = torch.relu(policy_entropy - 0.6 * max_entropy)  # Lower threshold
        
        # Total loss with stronger regularization
        loss = kl_loss + 0.5 * entropy_penalty  # Much stronger penalty
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping to prevent instability
        torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}, Entropy = {policy_entropy.item():.4f}")
    
    # Test the fixed model
    print("\n🧪 Testing aggressively fixed model...")
    model.eval()
    
    with torch.no_grad():
        p, v = model(x)
        p_probs = torch.softmax(p, dim=1)
        
    new_entropy = -(p_probs * torch.log(p_probs + 1e-8)).sum().item()
    print(f"New policy entropy: {new_entropy:.4f}")
    
    # Check if policy is more diverse
    legal_moves = list(board.legal_moves)
    move_probs = []
    
    for move in legal_moves:
        try:
            idx = move_to_index(board, move)
            prob = p_probs[0, idx].item()
            move_probs.append((move, prob))
        except:
            continue
    
    # Sort by probability
    move_probs.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nTop 10 moves after aggressive fixing:")
    for i, (move, prob) in enumerate(move_probs[:10]):
        print(f"{i+1:2d}. {board.san(move):<6}: {prob:.4f}")
    
    # Save the fixed model
    print("\n💾 Saving aggressively fixed model...")
    fixed_state = {
        'model': model.state_dict(),
        'fixed_policy': True,
        'aggressive_fix': True,
        'original_entropy': 8.42,  # From our analysis
        'new_entropy': new_entropy
    }
    
    torch.save(fixed_state, 'checkpoints/aggressive_fixed_policy.pt')
    print("✅ Aggressively fixed model saved to checkpoints/aggressive_fixed_policy.pt")
    
    if new_entropy < 5.0:  # Much lower than the original 8.42
        print("🎉 Policy diversity dramatically improved!")
    elif new_entropy < 6.5:
        print("✅ Policy diversity significantly improved!")
    else:
        print("⚠️  Policy still too uniform, may need architectural changes")

def move_to_index(board, move):
    """Helper function for move encoding"""
    from azchess.encoding import move_to_index
    return move_to_index(board, move)

if __name__ == "__main__":
    fix_model_aggressive()
