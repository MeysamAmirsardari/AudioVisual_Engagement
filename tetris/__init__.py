"""Real, deterministic Tetris — playable by the subject (left/right keys), with the
behaviour recorded so any game replays bit-for-bit."""
from .tetris_game import COLORS, PIECE_ID, SHAPES, TetrisGame

__all__ = ["TetrisGame", "COLORS", "PIECE_ID", "SHAPES"]
