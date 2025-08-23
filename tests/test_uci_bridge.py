import asyncio

import chess

from azchess.engines.uci_bridge import SynchronousUCIClient

def test_event_loop_restored_after_synchronous_client():
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        client = SynchronousUCIClient("/nonexistent/engine", {})
        client.start_engine()
        board = chess.Board()
        client.get_move(board)
        client.analyze_position(board)
        client.stop_engine()
        assert asyncio.get_event_loop() is loop
        async def dummy():
            return 123
        result = loop.run_until_complete(dummy())
        assert result == 123
    finally:
        loop.close()
        asyncio.set_event_loop(None)
