#!/usr/bin/env python3

"""
Fix Model Script - Retrain the model to resolve uniform policy outputs
"""

import torch
import chess
import numpy as np
from azchess.model.resnet import PolicyValueNet
from azchess.config import Config
from azchess.encoding import encode_board
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import logging

def fix_model():
    print("🔧 Fixing model with uniform policy issue...")
    
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
    
    # Now retrain the policy head with proper initialization
    print("\n🔄 Retraining policy head...")
    
    # Freeze all layers except policy head
    for name, param in model.named_parameters():
        if 'policy' not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
            print(f"Training: {name}")
    
    # Reinitialize policy head with proper weights
    model._init_policy_head()
    
    # Create synthetic training data to encourage diversity
    # We'll use a simple approach: encourage the model to prefer different moves
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    scheduler = CosineAnnealingLR(optimizer, T_max=100)
    
    # Training loop
    model.train()
    for epoch in range(50):
        optimizer.zero_grad()
        
        # Create a batch of positions
        positions = []
        targets = []
        
        for i in range(8):
            # Create different board positions
            board = chess.Board()
            if i > 0:
                # Make some random moves to create variety
                for _ in range(i):
                    legal_moves = list(board.legal_moves)
                    if legal_moves:
                        board.push(np.random.choice(legal_moves))
            
            x = torch.from_numpy(encode_board(board)).unsqueeze(0)
            positions.append(x)
            
            # Create target policy that encourages diversity
            # We'll use a simple approach: encourage the model to prefer different moves
            target = torch.zeros(1, 4672)
            legal_moves = list(board.legal_moves)
            
            if legal_moves:
                # Assign higher probability to "good" moves
                for move in legal_moves:
                    idx = move_to_index(board, move)
                    if idx < len(target[0]):
                        # Simple heuristic scoring
                        score = 0.1
                        
                        # Bonus for center control
                        if move.to_square in [chess.E4, chess.E5, chess.D4, chess.D5]:
                            score += 0.3
                        
                        # Bonus for pawn advances
                        if board.piece_at(move.from_square) and board.piece_at(move.from_square).piece_type == chess.PAWN:
                            score += 0.2
                        
                        # Bonus for developing pieces
                        if board.piece_at(move.from_square) and board.piece_at(move.from_square).piece_type in [chess.KNIGHT, chess.BISHOP]:
                            score += 0.2
                        
                        target[0, idx] = score
                
                # Normalize to probabilities
                target = torch.softmax(target, dim=1)
            
            targets.append(target)
        
        # Forward pass
        batch_x = torch.cat(positions, dim=0)
        p, v = model(batch_x)
        
        # Policy loss with diversity encouragement
        p_probs = torch.softmax(p, dim=1)
        
        # KL divergence loss to encourage diversity
        target_probs = torch.cat(targets, dim=0)
        kl_loss = torch.nn.functional.kl_div(
            torch.log(p_probs + 1e-8), 
            target_probs, 
            reduction='batchmean'
        )
        
        # Entropy regularization to prevent uniform outputs
        policy_entropy = -(p_probs * torch.log(p_probs + 1e-8)).sum(dim=1).mean()
        max_entropy = torch.log(torch.tensor(p_probs.shape[1], dtype=torch.float32))
        entropy_penalty = torch.relu(policy_entropy - 0.8 * max_entropy)
        
        # Total loss
        loss = kl_loss + 0.1 * entropy_penalty
        
        # Backward pass
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}, Entropy = {policy_entropy.item():.4f}")
    
    # Test the fixed model
    print("\n🧪 Testing fixed model...")
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
    
    print(f"\nTop 5 moves after fixing:")
    for i, (move, prob) in enumerate(move_probs[:5]):
        print(f"{i+1}. {board.san(move)}: {prob:.4f}")
    
    # Save the fixed model
    print("\n💾 Saving fixed model...")
    fixed_state = {
        'model': model.state_dict(),
        'fixed_policy': True,
        'original_entropy': 8.42,  # From our analysis
        'new_entropy': new_entropy
    }
    
    torch.save(fixed_state, 'checkpoints/fixed_policy.pt')
    print("✅ Fixed model saved to checkpoints/fixed_policy.pt")
    
    if new_entropy < 6.0:  # Much lower than the original 8.42
        print("🎉 Policy diversity significantly improved!")
    else:
        print("⚠️  Policy still too uniform, may need more training")

def move_to_index(board, move):
    """Helper function for move encoding"""
    from azchess.encoding import move_to_index
    return move_to_index(board, move)

if __name__ == "__main__":
    fix_model()
