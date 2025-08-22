# Benchmark Configuration System
"""
Configuration management for the Matrix0 benchmark system.
Handles UCI engine configurations, test scenarios, and performance settings.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class UCIEngineConfig:
    """Configuration for a UCI-compliant chess engine."""
    name: str
    command: str
    working_dir: Optional[str] = None
    options: Dict[str, Any] = None

    def __post_init__(self):
        if self.options is None:
            self.options = {}


@dataclass
class TestScenario:
    """Configuration for a single benchmark test scenario."""
    name: str
    engine_config: UCIEngineConfig
    model_checkpoint: str
    num_games: int = 100
    time_control: str = "60+0.6"  # UCI time control format
    opening_book: Optional[str] = None
    max_moves: int = 200
    concurrency: int = 1  # Number of games to run in parallel
    random_openings: bool = True
    opening_plies: int = 8


@dataclass
class PerformanceConfig:
    """Configuration for performance monitoring."""
    track_cpu: bool = True
    track_memory: bool = True
    track_gpu: bool = True
    sample_interval: float = 0.1  # seconds
    log_system_load: bool = True


@dataclass
class BenchmarkConfig:
    """Main benchmark configuration."""
    name: str
    description: str
    output_dir: str = "benchmarks/results"
    scenarios: List[TestScenario] = None
    performance_config: PerformanceConfig = None
    statistical_config: Dict[str, Any] = None

    def __post_init__(self):
        if self.scenarios is None:
            self.scenarios = []
        if self.performance_config is None:
            self.performance_config = PerformanceConfig()
        if self.statistical_config is None:
            self.statistical_config = {
                "confidence_level": 0.95,
                "min_games_for_stats": 30,
                "calculate_elo": True,
                "elo_k_factor": 32
            }


class ConfigManager:
    """Manages benchmark configurations."""

    @staticmethod
    def load_config(config_path: str) -> BenchmarkConfig:
        """Load benchmark configuration from YAML file."""
        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)

        # Parse UCI engines
        engines = {}
        if 'engines' in data:
            for engine_name, engine_data in data['engines'].items():
                engines[engine_name] = UCIEngineConfig(
                    name=engine_name,
                    command=engine_data['command'],
                    working_dir=engine_data.get('working_dir'),
                    options=engine_data.get('options', {})
                )

        # Parse scenarios
        scenarios = []
        if 'scenarios' in data:
            for scenario_data in data['scenarios']:
                engine_config = engines.get(scenario_data['engine'])
                if not engine_config:
                    raise ValueError(f"Engine '{scenario_data['engine']}' not found in engines section")

                scenarios.append(TestScenario(
                    name=scenario_data['name'],
                    engine_config=engine_config,
                    model_checkpoint=scenario_data['model_checkpoint'],
                    num_games=scenario_data.get('num_games', 100),
                    time_control=scenario_data.get('time_control', '60+0.6'),
                    max_moves=scenario_data.get('max_moves', 200),
                    concurrency=scenario_data.get('concurrency', 1),
                    random_openings=scenario_data.get('random_openings', True),
                    opening_plies=scenario_data.get('opening_plies', 8)
                ))

        # Parse performance config
        perf_config = PerformanceConfig()
        if 'performance' in data:
            perf_data = data['performance']
            perf_config = PerformanceConfig(
                track_cpu=perf_data.get('track_cpu', True),
                track_memory=perf_data.get('track_memory', True),
                track_gpu=perf_data.get('track_gpu', True),
                sample_interval=perf_data.get('sample_interval', 0.1),
                log_system_load=perf_data.get('log_system_load', True)
            )

        return BenchmarkConfig(
            name=data.get('name', 'Matrix0 Benchmark'),
            description=data.get('description', ''),
            output_dir=data.get('output_dir', 'benchmarks/results'),
            scenarios=scenarios,
            performance_config=perf_config,
            statistical_config=data.get('statistics', {})
        )

    @staticmethod
    def save_config(config: BenchmarkConfig, output_path: str):
        """Save benchmark configuration to YAML file."""
        data = asdict(config)

        # Convert complex objects to dictionaries
        for scenario in data['scenarios']:
            scenario['engine_config'] = asdict(scenario['engine_config'])
        data['performance_config'] = asdict(data['performance_config'])

        with open(output_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, indent=2)

    @staticmethod
    def create_default_config() -> BenchmarkConfig:
        """Create a default benchmark configuration."""
        stockfish_config = UCIEngineConfig(
            name="Stockfish",
            command="stockfish",
            options={"Threads": "4", "Hash": "512"}
        )

        lc0_config = UCIEngineConfig(
            name="lc0",
            command="lc0",
            options={"Threads": "4", "NNCacheSize": "2000000"}
        )

        scenarios = [
            TestScenario(
                name="Stockfish_ELO_2000",
                engine_config=stockfish_config,
                model_checkpoint="checkpoints/v2_base.pt",
                num_games=50,
                time_control="60+0.6",
                concurrency=2
            ),
            TestScenario(
                name="lc0_Medium",
                engine_config=lc0_config,
                model_checkpoint="checkpoints/v2_base.pt",
                num_games=50,
                time_control="30+0.3",
                concurrency=2
            )
        ]

        return BenchmarkConfig(
            name="Matrix0 Standard Benchmark",
            description="Standard benchmark suite against Stockfish and lc0",
            scenarios=scenarios
        )


# Predefined engine configurations for common engines
PREDEFINED_ENGINES = {
    "stockfish": UCIEngineConfig(
        name="Stockfish",
        command="stockfish",
        options={
            "Threads": "4",
            "Hash": "512",
            "Skill Level": "20"
        }
    ),

    "lc0": UCIEngineConfig(
        name="lc0",
        command="lc0",
        options={
            "Threads": "4",
            "NNCacheSize": "2000000",
            "MinibatchSize": "32"
        }
    ),

    "komodo": UCIEngineConfig(
        name="Komodo",
        command="komodo",
        options={
            "Threads": "4",
            "Hash": "512"
        }
    )
}
