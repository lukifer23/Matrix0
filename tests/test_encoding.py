
import unittest
import chess
import numpy as np

from azchess.encoding import MoveEncoder, move_to_index, encode_board

class TestMoveEncoding(unittest.TestCase):
    """
    Test suite for the move encoding system from azchess.encoding.
    This combines tests from the original test_encoding.py and test_move_encoding.py.
    """

    def setUp(self):
        """Set up test fixtures."""
        self.encoder = MoveEncoder()
        self.board = chess.Board()

    # --- Tests from original test_encoding.py ---

    def test_castling_indices_different(self):
        """Tests that kingside and queenside castling produce different indices."""
        b = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
        k_castle = chess.Move.from_uci("e1g1")
        q_castle = chess.Move.from_uci("e1c1")
        idx_k = self.encoder.encode_move(b, k_castle)
        idx_q = self.encoder.encode_move(b, q_castle)
        self.assertNotEqual(idx_k, idx_q, "Kingside and Queenside castling should have different indices")

    def test_en_passant_maps_like_capture(self):
        """Tests that an en passant move is encoded as a valid index."""
        b = chess.Board("8/8/8/3pP3/8/8/8/8 w - d6 0 2")
        ep_move = chess.Move.from_uci("e5d6")
        idx = self.encoder.encode_move(b, ep_move)
        self.assertTrue(0 <= idx < 4672, "En passant move index should be within the valid range")

    def test_promotion_under_and_queen(self):
        """Tests that underpromotions and queen promotions have different indices."""
        b = chess.Board("8/P7/8/8/8/8/8/4k2K w - - 0 1")
        knight_promo = chess.Move.from_uci("a7a8n")
        queen_promo = chess.Move.from_uci("a7a8q")
        idx_knight = self.encoder.encode_move(b, knight_promo)
        idx_queen = self.encoder.encode_move(b, queen_promo)
        self.assertTrue(0 <= idx_knight < 4672)
        self.assertTrue(0 <= idx_queen < 4672)
        self.assertNotEqual(idx_knight, idx_queen, "Underpromotion and queen promotion should have different indices")

    # --- Tests adapted from test_move_encoding.py ---

    def test_basic_move_encoding_decoding(self):
        """Test basic move encoding and decoding."""
        move = chess.Move.from_uci("e2e4")
        self.board.push(move)
        knight_move = chess.Move.from_uci("g8f6")

        action_idx = self.encoder.encode_move(self.board, knight_move)
        self.assertTrue(0 <= action_idx < 4672)

        decoded_move = self.encoder.decode_move(self.board, action_idx)
        self.assertEqual(knight_move, decoded_move)

    def test_all_legal_moves_are_encodable_and_decodable(self):
        """Ensure all legal moves on a board can be encoded and decoded back."""
        b = chess.Board("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1") # Kiwipete position
        for move in b.legal_moves:
            with self.subTest(move=move.uci()):
                action_idx = self.encoder.encode_move(b, move)
                decoded_move = self.encoder.decode_move(b, action_idx)
                self.assertEqual(move, decoded_move)

    def test_legal_actions_mask(self):
        """Test legal actions mask generation."""
        legal_mask = self.encoder.get_legal_actions(self.board)
        self.assertEqual(legal_mask.shape, (4672,))
        self.assertEqual(legal_mask.dtype, bool)
        # Starting position has 20 legal moves
        self.assertEqual(legal_mask.sum(), 20)

        # After e2e4
        self.board.push(chess.Move.from_uci("e2e4"))
        legal_mask_2 = self.encoder.get_legal_actions(self.board)
        self.assertEqual(legal_mask_2.sum(), 20)

    def test_encoding_validation_on_random_boards(self):
        """Test that encoding validation works correctly on random boards."""
        for _ in range(10): # Test on 10 random boards
            b = chess.Board()
            # Play some random moves
            for _ in range(np.random.randint(1, 50)):
                if b.is_game_over():
                    break
                legal_moves = list(b.legal_moves)
                b.push(np.random.choice(legal_moves))
            
            if not b.is_game_over():
                self.assertTrue(self.encoder.validate_encoding(b))

    def test_board_encoding(self):
        """Test board encoding functionality."""
        encoded = encode_board(self.board)
        self.assertEqual(encoded.shape, (19, 8, 8))
        self.assertEqual(encoded.dtype, np.float32)

        # White pawns are on the second rank (index 6 from white's perspective)
        self.assertTrue(np.all(encoded[0][6, :] == 1.0))
        # Black pawns are on the seventh rank (index 1)
        self.assertTrue(np.all(encoded[6][1, :] == 1.0))
        
        # Side to move is white (1.0)
        self.assertTrue(np.all(encoded[12] == 1.0))
        
        # Castling rights are present
        self.assertTrue(np.all(encoded[13] == 1.0)) # White kingside
        self.assertTrue(np.all(encoded[14] == 1.0)) # White queenside
        self.assertTrue(np.all(encoded[15] == 1.0)) # Black kingside
        self.assertTrue(np.all(encoded[16] == 1.0)) # Black queenside

if __name__ == '__main__':
    unittest.main(verbosity=2)
