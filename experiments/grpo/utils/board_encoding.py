import torch
import chess

def board_to_tensor(board: chess.Board) -> torch.Tensor:
    """Convert a chess.Board to a (1, 19, 8, 8) tensor.

    Channels are ordered as:
        0-11: piece planes (white then black for P,N,B,R,Q,K)
        12: side to move (1 if white, 0 if black)
        13-16: castling rights (W-K, W-Q, B-K, B-Q)
        17: en passant target square
        18: halfmove clock normalized to [0,1] with cap at 100
    """
    channels = []

    # Piece channels
    for piece_type in range(1, 7):  # 1-6: pawn, knight, bishop, rook, queen, king
        white_channel = torch.zeros(8, 8)
        black_channel = torch.zeros(8, 8)
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.piece_type == piece_type:
                row, col = divmod(square, 8)
                if piece.color == chess.WHITE:
                    white_channel[row, col] = 1.0
                else:
                    black_channel[row, col] = 1.0
        channels.extend([white_channel, black_channel])

    # Side to move
    side_to_move = torch.ones(8, 8) if board.turn == chess.WHITE else torch.zeros(8, 8)
    channels.append(side_to_move)

    # Castling rights
    castling_channels = [torch.zeros(8, 8) for _ in range(4)]
    if board.has_kingside_castling_rights(chess.WHITE):
        castling_channels[0].fill_(1.0)
    if board.has_queenside_castling_rights(chess.WHITE):
        castling_channels[1].fill_(1.0)
    if board.has_kingside_castling_rights(chess.BLACK):
        castling_channels[2].fill_(1.0)
    if board.has_queenside_castling_rights(chess.BLACK):
        castling_channels[3].fill_(1.0)
    channels.extend(castling_channels)

    # En passant square
    en_passant = torch.zeros(8, 8)
    if board.ep_square is not None:
        row, col = divmod(board.ep_square, 8)
        en_passant[row, col] = 1.0
    channels.append(en_passant)

    # Halfmove clock (normalized)
    halfmove = torch.full((8, 8), min(board.halfmove_clock / 100.0, 1.0))
    channels.append(halfmove)

    return torch.stack(channels, dim=0).unsqueeze(0)
