#!/usr/bin/env python3
"""
SSL Algorithms Demonstration
Showcases the advanced self-supervised learning algorithms for Matrix0.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from azchess.ssl_algorithms import ChessSSLAlgorithms
from azchess.encoding import encode_board
from azchess.config import Config


def create_demo_board():
    """Create a demonstration board with tactical elements."""
    # Create a board with pins, threats, and forks
    board = torch.zeros(1, 19, 8, 8)

    # White pieces
    board[0, 0, 6, 0] = 1  # White pawn at a2
    board[0, 0, 6, 1] = 1  # White pawn at b2
    board[0, 0, 6, 3] = 1  # White pawn at d2
    board[0, 0, 6, 4] = 1  # White pawn at e2
    board[0, 0, 6, 5] = 1  # White pawn at f2
    board[0, 0, 6, 6] = 1  # White pawn at g2
    board[0, 0, 6, 7] = 1  # White pawn at h2

    board[0, 1, 7, 1] = 1  # White knight at b1
    board[0, 1, 7, 6] = 1  # White knight at g1
    board[0, 2, 7, 2] = 1  # White bishop at c1
    board[0, 2, 7, 5] = 1  # White bishop at f1
    board[0, 3, 7, 0] = 1  # White rook at a1
    board[0, 3, 7, 7] = 1  # White rook at h1
    board[0, 4, 7, 3] = 1  # White queen at d1
    board[0, 5, 7, 4] = 1  # White king at e1

    # Black pieces
    board[0, 6, 1, 0] = 1  # Black pawn at a7
    board[0, 6, 1, 1] = 1  # Black pawn at b7
    board[0, 6, 1, 2] = 1  # Black pawn at c7
    board[0, 6, 1, 3] = 1  # Black pawn at d7
    board[0, 6, 1, 4] = 1  # Black pawn at e7
    board[0, 6, 1, 5] = 1  # Black pawn at f7
    board[0, 6, 1, 6] = 1  # Black pawn at g7
    board[0, 6, 1, 7] = 1  # Black pawn at h7

    board[0, 7, 0, 1] = 1  # Black knight at b8
    board[0, 7, 0, 6] = 1  # Black knight at g8
    board[0, 8, 0, 2] = 1  # Black bishop at c8
    board[0, 8, 0, 5] = 1  # Black bishop at f8
    board[0, 9, 0, 0] = 1  # Black rook at a8
    board[0, 9, 0, 7] = 1  # Black rook at h8
    board[0, 10, 0, 3] = 1 # Black queen at d8
    board[0, 11, 0, 4] = 1 # Black king at e8

    # Set side to move (White)
    board[0, 12, 0, 0] = 1

    return board


def demo_piece_recognition():
    """Demonstrate piece recognition SSL task."""
    print("=" * 60)
    print("Piece Recognition SSL Task Demo")
    print("=" * 60)

    ssl_alg = ChessSSLAlgorithms()
    board = create_demo_board()

    print("Analyzing piece recognition on demonstration board...")

    # Get piece recognition targets
    piece_targets = ssl_alg._create_piece_targets(board)

    # Count pieces by type
    piece_counts = {}
    for piece_type in range(13):  # 12 piece types + empty squares
        count = (piece_targets[0, piece_type] == 1).sum().item()
        if count > 0:
            if piece_type == 12:
                piece_name = "Empty squares"
            else:
                piece_names = ["Pawn", "Knight", "Bishop", "Rook", "Queen", "King"] * 2
                colors = ["White"] * 6 + ["Black"] * 6
                piece_name = f"{colors[piece_type]} {piece_names[piece_type]}"
            piece_counts[piece_name] = count

    print("Piece recognition results:")
    for piece_name, count in piece_counts.items():
        print(f"  {piece_name}: {count}")

    total_pieces = sum(count for name, count in piece_counts.items() if "Empty" not in name)
    total_empty = piece_counts.get("Empty squares", 0)
    print(f"\nTotal pieces detected: {total_pieces}")
    print(f"Empty squares: {total_empty}")
    print(f"Board total: {total_pieces + total_empty}")

    print("\n✅ Piece recognition demonstration complete")


def demo_threat_detection():
    """Demonstrate threat detection SSL task."""
    print("\n" + "=" * 60)
    print("Threat Detection SSL Task Demo")
    print("=" * 60)

    ssl_alg = ChessSSLAlgorithms()
    board = create_demo_board()

    print("Analyzing threat patterns...")

    threat_map = ssl_alg.detect_threats_batch(board)

    threatened_squares = (threat_map > 0).sum().item()
    print(f"Squares under attack: {threatened_squares}")

    # Show threat heatmap
    print("\nThreat heatmap (1 = threatened):")
    for row in range(8):
        row_str = ""
        for col in range(8):
            threat = threat_map[0, 0, row, col].item()
            row_str += "1" if threat > 0 else "0"
        print(f"  {8-row}: {row_str}")

    print("\n✅ Threat detection demonstration complete")


def demo_pin_detection():
    """Demonstrate pin detection SSL task."""
    print("\n" + "=" * 60)
    print("Pin Detection SSL Task Demo")
    print("=" * 60)

    ssl_alg = ChessSSLAlgorithms()

    # Create a board with a pin
    board = torch.zeros(1, 19, 8, 8)

    # White king at e1
    board[0, 5, 7, 4] = 1
    # White queen at d1 (pinned by black bishop)
    board[0, 4, 7, 3] = 1
    # Black bishop at g4
    board[0, 8, 3, 6] = 1

    # White to move
    board[0, 12, 0, 0] = 1

    print("Analyzing pin detection on tactical position...")
    print("Position: White king e1, queen d1, Black bishop g4")

    pin_map = ssl_alg.detect_pins_batch(board)
    pinned_pieces = (pin_map > 0).sum().item()

    print(f"Pinned pieces detected: {pinned_pieces}")

    if pinned_pieces > 0:
        print("\nPin locations:")
        for row in range(8):
            for col in range(8):
                if pin_map[0, 0, row, col].item() > 0:
                    square = chr(ord('a') + col) + str(8 - row)
                    print(f"  Piece pinned at {square}")

    print("\n✅ Pin detection demonstration complete")


def demo_square_control():
    """Demonstrate square control SSL task."""
    print("\n" + "=" * 60)
    print("Square Control SSL Task Demo")
    print("=" * 60)

    ssl_alg = ChessSSLAlgorithms()
    board = create_demo_board()

    print("Analyzing square control...")

    control_map = ssl_alg.calculate_square_control_batch(board)

    white_control = (control_map > 0).sum().item()
    black_control = (control_map < 0).sum().item()
    contested = (control_map == 0).sum().item()

    print(f"White controls: {white_control} squares")
    print(f"Black controls: {black_control} squares")
    print(f"Contested/neutral: {contested} squares")

    # Show control heatmap
    print("\nControl heatmap (W=White, B=Black, -=Neutral):")
    for row in range(8):
        row_str = ""
        for col in range(8):
            control = control_map[0, 0, row, col].item()
            if control > 0:
                row_str += "W"
            elif control < 0:
                row_str += "B"
            else:
                row_str += "-"
        print(f"  {8-row}: {row_str}")

    print("\n✅ Square control demonstration complete")


def demo_enhanced_ssl_targets():
    """Demonstrate the enhanced SSL target creation."""
    print("\n" + "=" * 60)
    print("Enhanced SSL Targets Demo")
    print("=" * 60)

    ssl_alg = ChessSSLAlgorithms()
    board = create_demo_board()

    print("Creating enhanced SSL targets for all tasks...")

    ssl_tasks = ['piece', 'threat', 'pin', 'fork', 'control']
    ssl_targets = ssl_alg.create_enhanced_ssl_targets(board, ssl_tasks)

    print(f"Created {len(ssl_targets)} SSL target types:")
    for task_name, target_tensor in ssl_targets.items():
        shape = target_tensor.shape
        print(f"  {task_name}: shape {shape}")

        # Calculate some statistics
        if task_name == 'threat':
            threatened = (target_tensor > 0).sum().item()
            print(f"    Squares under threat: {threatened}")
        elif task_name == 'control':
            white_control = (target_tensor > 0).sum().item()
            black_control = (target_tensor < 0).sum().item()
            print(f"    White control: {white_control}, Black control: {black_control}")

    print("\n✅ Enhanced SSL targets demonstration complete")


def demo_model_integration():
    """Demonstrate SSL integration with the main model."""
    print("\n" + "=" * 60)
    print("SSL Model Integration Demo")
    print("=" * 60)

    try:
        from azchess.model import PolicyValueNet, NetConfig

        # Create a minimal config for demo
        cfg_dict = {
            'model': {
                'planes': 19,
                'channels': 32,  # Small for demo
                'blocks': 1,     # Small for demo
                'policy_size': 4672,
                'attention_heads': 2,
                'self_supervised': True,
                'ssl_tasks': ['piece', 'threat', 'pin', 'fork', 'control']
            }
        }

        print("Creating model with SSL integration...")
        model_config = NetConfig(**cfg_dict['model'])
        model = PolicyValueNet(model_config)

        board = create_demo_board()

        print("Testing SSL target creation through model...")
        ssl_targets = model.create_ssl_targets(board)

        if ssl_targets:
            print(f"Model created {len(ssl_targets)} SSL targets")
            for task_name, target in ssl_targets.items():
                print(f"  {task_name}: {target.shape}")
        else:
            print("No SSL targets created (model may not have SSL enabled)")

        print("\n✅ SSL model integration demonstration complete")

    except Exception as e:
        print(f"Model integration demo failed: {e}")
        print("This may be due to missing dependencies or model configuration issues")


def main():
    """Run all SSL algorithm demonstrations."""
    print("Matrix0 SSL Algorithms Demonstration")
    print("=" * 60)
    print("This demo showcases the advanced self-supervised learning algorithms")
    print("that enhance Matrix0's understanding of chess positions.")
    print()

    # Run all demos
    demo_piece_recognition()
    demo_threat_detection()
    demo_pin_detection()
    demo_square_control()
    demo_enhanced_ssl_targets()
    demo_model_integration()

    print("\n" + "=" * 60)
    print("🎉 All SSL algorithm demonstrations completed!")
    print("=" * 60)
    print("\nThe SSL algorithms provide:")
    print("• Piece recognition: Identify all pieces and empty squares")
    print("• Threat detection: Find squares under attack by enemy pieces")
    print("• Pin detection: Identify pinned pieces and their constraints")
    print("• Fork detection: Find tactical opportunities for forking pieces")
    print("• Square control: Determine which player controls each square")
    print("\nSSL Benefits:")
    print("• Enhanced positional understanding")
    print("• Better tactical recognition")
    print("• Improved strategic planning")
    print("• More accurate move evaluation")
    print("• Self-supervised learning without human labels")
    print("\nThese algorithms work together to give Matrix0 a deeper")
    print("understanding of chess positions beyond just policy and value!")
    print("\nSSL algorithms are now integrated into Matrix0's training pipeline! 🧠")


if __name__ == "__main__":
    main()
