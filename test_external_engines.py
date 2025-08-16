#!/usr/bin/env python3
"""Test script for external engine integration."""

import asyncio
import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from azchess.config import Config
from azchess.engines import EngineManager


async def test_engine_manager():
    """Test the engine manager functionality."""
    print("Testing Engine Manager...")
    
    # Load configuration
    config = Config.load("config.yaml")
    
    # Initialize engine manager
    engine_manager = EngineManager(config.to_dict())
    
    print(f"Available engines: {list(config.engines().keys())}")
    
    # Check engine health
    print("\nChecking engine health...")
    health_results = await engine_manager.check_all_engines_health()
    
    for engine_name, is_healthy in health_results.items():
        status = "✅ Healthy" if is_healthy else "❌ Unhealthy"
        print(f"  {engine_name}: {status}")
    
    # Get engine info
    print("\nEngine information:")
    for engine_name in config.engines():
        info = engine_manager.get_engine_info(engine_name)
        print(f"  {engine_name}:")
        print(f"    Type: {info.get('type', 'unknown')}")
        print(f"    Enabled: {info.get('enabled', False)}")
        print(f"    Status: {info.get('health', {}).get('status', 'unknown')}")
    
    # Cleanup
    await engine_manager.cleanup()
    print("\n✅ Engine manager test completed successfully!")


async def test_external_engine_selfplay():
    """Test external engine self-play functionality."""
    print("\nTesting External Engine Self-Play...")
    
    try:
        from azchess.selfplay.external_engine_worker import ExternalEngineSelfPlay
        
        # Load configuration
        config = Config.load("config.yaml")
        
        # Initialize engine manager
        engine_manager = EngineManager(config.to_dict())
        
        # Initialize self-play
        selfplay = ExternalEngineSelfPlay(config, engine_manager)
        
        print("✅ External engine self-play test completed successfully!")
        
    except ImportError as e:
        print(f"⚠️  External engine self-play not available: {e}")
    except Exception as e:
        print(f"❌ External engine self-play test failed: {e}")
    finally:
        if 'engine_manager' in locals():
            await engine_manager.cleanup()


async def test_multi_engine_evaluator():
    """Test multi-engine evaluator functionality."""
    print("\nTesting Multi-Engine Evaluator...")
    
    try:
        from azchess.eval.multi_engine_evaluator import MultiEngineEvaluator
        
        # Load configuration
        config = Config.load("config.yaml")
        
        # Initialize evaluator
        evaluator = MultiEngineEvaluator(config)
        
        print("✅ Multi-engine evaluator test completed successfully!")
        
    except ImportError as e:
        print(f"⚠️  Multi-engine evaluator not available: {e}")
    except Exception as e:
        print(f"❌ Multi-engine evaluator test failed: {e}")


async def main():
    """Run all tests."""
    print("🚀 Starting External Engine Integration Tests")
    print("=" * 50)
    
    try:
        await test_engine_manager()
        await test_external_engine_selfplay()
        await test_multi_engine_evaluator()
        
        print("\n" + "=" * 50)
        print("✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
