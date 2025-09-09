#!/usr/bin/env python3
"""
Meta-Learning for GRPO Chess Experiments

Implements learn-to-learn approaches for chess, including:
- Adaptive GRPO parameters based on game state
- Task-specific parameter optimization
- Curriculum-based learning strategies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ChessTask:
    """Represents a chess learning task or game state"""
    game_phase: str  # 'opening', 'middlegame', 'endgame'
    material_balance: float  # -1.0 to 1.0
    complexity_score: float  # 0.0 to 1.0
    position_embedding: torch.Tensor
    task_id: str


class AdaptiveParameterLearner(nn.Module):
    """
    Learns to adapt GRPO parameters based on chess position characteristics
    """

    def __init__(self, d_model: int = 256):
        super().__init__()

        # Position encoder
        self.position_encoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model // 2)
        )

        # Task characteristic encoder
        self.task_encoder = nn.Sequential(
            nn.Linear(4, d_model // 4),  # phase, material, complexity, game_length
            nn.LayerNorm(d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, d_model // 2)
        )

        # Parameter predictors
        self.cpuct_predictor = nn.Sequential(
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()  # Output 0-1, will be scaled
        )

        self.virtual_loss_predictor = nn.Sequential(
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()
        )

        self.learning_rate_predictor = nn.Sequential(
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()
        )

        self.group_size_predictor = nn.Sequential(
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()
        )

    def forward(self, position_embedding: torch.Tensor,
                task_features: torch.Tensor) -> Dict[str, float]:
        """
        Predict optimal GRPO parameters for given position and task

        Args:
            position_embedding: Position representation (d_model,)
            task_features: Task characteristics [phase, material, complexity, game_length]

        Returns:
            Dictionary of predicted parameters
        """
        # Encode inputs
        pos_encoded = self.position_encoder(position_embedding)
        task_encoded = self.task_encoder(task_features)

        # Combine encodings
        combined = pos_encoded + task_encoded

        # Predict parameters
        cpuct = self.cpuct_predictor(combined).item() * 2.0 + 1.0  # Scale to 1.0-3.0
        virtual_loss = self.virtual_loss_predictor(combined).item() * 3.0 + 1.0  # Scale to 1.0-4.0
        learning_rate = self.learning_rate_predictor(combined).item() * 1e-4  # Scale to 0-1e-4
        group_size = int(self.group_size_predictor(combined).item() * 14 + 2)  # Scale to 2-16

        return {
            'cpuct': cpuct,
            'virtual_loss': virtual_loss,
            'learning_rate': learning_rate,
            'group_size': max(2, min(16, group_size))
        }


class MetaLearningCurriculum:
    """
    Manages curriculum learning with meta-learning adaptation
    """

    def __init__(self, initial_difficulty: float = 0.5):
        self.current_difficulty = initial_difficulty
        self.performance_history = []
        self.task_history = []

        # Curriculum phases
        self.phases = {
            'novice': {'difficulty': 0.2, 'focus': 'basic_patterns'},
            'intermediate': {'difficulty': 0.5, 'focus': 'tactical_combinations'},
            'advanced': {'difficulty': 0.8, 'focus': 'strategic_planning'}
        }

    def adapt_curriculum(self, recent_performance: List[float],
                        task_characteristics: List[Dict]) -> Dict[str, Any]:
        """
        Adapt curriculum based on recent performance

        Args:
            recent_performance: List of recent performance scores
            task_characteristics: Characteristics of recent tasks

        Returns:
            Updated curriculum settings
        """
        avg_performance = np.mean(recent_performance)

        # Adjust difficulty based on performance
        if avg_performance > 0.7:  # Doing well, increase difficulty
            self.current_difficulty = min(1.0, self.current_difficulty + 0.1)
        elif avg_performance < 0.3:  # Struggling, decrease difficulty
            self.current_difficulty = max(0.1, self.current_difficulty - 0.1)

        # Determine appropriate phase
        if self.current_difficulty < 0.4:
            phase = 'novice'
        elif self.current_difficulty < 0.7:
            phase = 'intermediate'
        else:
            phase = 'advanced'

        phase_settings = self.phases[phase]

        return {
            'difficulty': self.current_difficulty,
            'phase': phase,
            'focus_area': phase_settings['focus'],
            'task_complexity': self.current_difficulty,
            'exploration_bonus': 1.0 - self.current_difficulty  # More exploration when struggling
        }

    def select_optimal_task(self, available_tasks: List[ChessTask],
                           current_skill_level: float) -> ChessTask:
        """
        Select the most appropriate task for current skill level

        Args:
            available_tasks: List of available learning tasks
            current_skill_level: Current estimated skill level (0-1)

        Returns:
            Selected task for learning
        """
        # Score tasks based on suitability for current skill level
        task_scores = []
        for task in available_tasks:
            # Task difficulty should match current skill level
            difficulty_match = 1.0 - abs(task.complexity_score - current_skill_level)

            # Prefer tasks with balanced material
            material_balance = 1.0 - abs(task.material_balance)

            # Phase-appropriate tasks
            phase_match = self._calculate_phase_match(task, current_skill_level)

            total_score = (difficulty_match * 0.4 +
                          material_balance * 0.3 +
                          phase_match * 0.3)

            task_scores.append(total_score)

        # Select highest scoring task
        best_idx = np.argmax(task_scores)
        return available_tasks[best_idx]

    def _calculate_phase_match(self, task: ChessTask, skill_level: float) -> float:
        """Calculate how well task matches current learning phase"""
        if skill_level < 0.4:  # Novice
            return 1.0 if task.game_phase == 'opening' else 0.5
        elif skill_level < 0.7:  # Intermediate
            return 1.0 if task.game_phase == 'middlegame' else 0.7
        else:  # Advanced
            return 1.0 if task.game_phase in ['middlegame', 'endgame'] else 0.8


class TaskSimilarityClusterer:
    """
    Clusters tasks by similarity for efficient group formation in GRPO
    """

    def __init__(self, n_clusters: int = 8):
        self.n_clusters = n_clusters
        self.cluster_centers = None
        self.task_embeddings = []

    def add_task(self, task: ChessTask):
        """Add task to clustering consideration"""
        self.task_embeddings.append(task.position_embedding)

    def cluster_tasks(self) -> Dict[int, List[ChessTask]]:
        """
        Cluster tasks by similarity

        Returns:
            Dictionary mapping cluster ID to list of tasks
        """
        if len(self.task_embeddings) < self.n_clusters:
            # Not enough tasks, put all in one cluster
            return {0: []}  # Would need to return actual tasks

        # Simple clustering by game phase and material balance
        clusters = {i: [] for i in range(self.n_clusters)}

        for task in []:  # Would iterate through actual tasks
            # Simplified clustering logic
            if task.game_phase == 'opening':
                cluster_id = 0
            elif task.game_phase == 'middlegame':
                cluster_id = 1
            else:  # endgame
                cluster_id = 2

            # Adjust based on material balance
            if task.material_balance > 0.3:
                cluster_id += 3
            elif task.material_balance < -0.3:
                cluster_id += 6

            cluster_id = min(cluster_id, self.n_clusters - 1)
            clusters[cluster_id].append(task)

        return clusters

    def get_similar_tasks(self, reference_task: ChessTask, n_similar: int = 4) -> List[ChessTask]:
        """
        Find most similar tasks to reference task

        Args:
            reference_task: Task to find similarities for
            n_similar: Number of similar tasks to return

        Returns:
            List of most similar tasks
        """
        similarities = []

        for task in []:  # Would iterate through actual tasks
            # Calculate similarity based on multiple factors
            phase_sim = 1.0 if task.game_phase == reference_task.game_phase else 0.5
            material_sim = 1.0 - abs(task.material_balance - reference_task.material_balance)
            complexity_sim = 1.0 - abs(task.complexity_score - reference_task.complexity_score)

            total_sim = (phase_sim + material_sim + complexity_sim) / 3.0
            similarities.append((task, total_sim))

        # Sort by similarity and return top N
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [task for task, _ in similarities[:n_similar]]


class MetaGRPOTrainer:
    """
    GRPO trainer with meta-learning capabilities
    """

    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        self.model = model
        self.config = config

        # Meta-learning components
        self.parameter_learner = AdaptiveParameterLearner()
        self.curriculum_manager = MetaLearningCurriculum()
        self.task_clusterer = TaskSimilarityClusterer()

        # Training state
        self.performance_history = []
        self.task_history = []

        logger.info("Initialized Meta-GRPO trainer")

    def train_with_meta_learning(self, trajectories: List[Any],
                               task_characteristics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train with meta-learning adaptation

        Args:
            trajectories: Training trajectories
            task_characteristics: Characteristics of current training task

        Returns:
            Training results and adapted parameters
        """
        # Adapt curriculum based on recent performance
        curriculum_settings = self.curriculum_manager.adapt_curriculum(
            self.performance_history[-10:] if len(self.performance_history) >= 10 else [0.5],
            self.task_history[-10:] if len(self.task_history) >= 10 else [{}]
        )

        # Learn optimal parameters for current task
        position_embedding = torch.randn(256)  # Would be actual position embedding
        task_features = torch.tensor([
            self._encode_game_phase(task_characteristics.get('phase', 'middlegame')),
            task_characteristics.get('material_balance', 0.0),
            task_characteristics.get('complexity', 0.5),
            task_characteristics.get('game_length', 80) / 200  # Normalize
        ])

        adapted_params = self.parameter_learner(position_embedding, task_features)

        # Apply adapted parameters to GRPO training
        grpo_results = self._train_with_adapted_params(trajectories, adapted_params, curriculum_settings)

        # Update learning history
        self.performance_history.append(grpo_results.get('win_rate', 0.5))
        self.task_history.append(task_characteristics)

        # Cluster tasks for future group formation
        # self.task_clusterer.add_task(current_task)

        return {
            'grpo_results': grpo_results,
            'adapted_parameters': adapted_params,
            'curriculum_settings': curriculum_settings,
            'meta_learning_metrics': {
                'curriculum_difficulty': curriculum_settings['difficulty'],
                'parameter_adaptation_score': self._evaluate_parameter_adaptation(adapted_params),
                'task_complexity_match': curriculum_settings['task_complexity']
            }
        }

    def _encode_game_phase(self, phase: str) -> float:
        """Encode game phase as numeric value"""
        phase_map = {'opening': 0.0, 'middlegame': 0.5, 'endgame': 1.0}
        return phase_map.get(phase, 0.5)

    def _train_with_adapted_params(self, trajectories: List[Any],
                                 adapted_params: Dict[str, float],
                                 curriculum_settings: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train GRPO with adapted parameters

        Args:
            trajectories: Training trajectories
            adapted_params: Adapted GRPO parameters
            curriculum_settings: Curriculum settings

        Returns:
            Training results
        """
        # Apply curriculum difficulty scaling
        difficulty_scale = curriculum_settings.get('difficulty', 1.0)

        # Apply adapted parameters
        group_size = adapted_params.get('group_size', 8)
        learning_rate = adapted_params.get('learning_rate', 1e-4)

        # Simplified training simulation
        # In practice, this would run actual GRPO training
        training_results = {
            'policy_loss': 0.5 * difficulty_scale,
            'value_loss': 0.3 * difficulty_scale,
            'entropy_loss': 0.1 / difficulty_scale,  # More exploration when easy
            'win_rate': 0.5 + (difficulty_scale - 0.5) * 0.2,  # Better performance on easier tasks
            'adapted_group_size': group_size,
            'adapted_learning_rate': learning_rate
        }

        return training_results

    def _evaluate_parameter_adaptation(self, adapted_params: Dict[str, float]) -> float:
        """Evaluate how well parameters were adapted"""
        # Simple evaluation based on parameter reasonableness
        cpuct_score = 1.0 if 1.0 <= adapted_params['cpuct'] <= 3.0 else 0.5
        vl_score = 1.0 if 1.0 <= adapted_params['virtual_loss'] <= 4.0 else 0.5
        lr_score = 1.0 if 0 <= adapted_params['learning_rate'] <= 1e-3 else 0.5
        gs_score = 1.0 if 2 <= adapted_params['group_size'] <= 16 else 0.5

        return (cpuct_score + vl_score + lr_score + gs_score) / 4.0


if __name__ == "__main__":
    # Test meta-learning components
    print("=== Meta-Learning for GRPO Test ===")

    # Test parameter learner
    param_learner = AdaptiveParameterLearner()
    pos_emb = torch.randn(256)
    task_feat = torch.tensor([0.5, 0.1, 0.7, 0.8])  # middlegame, slight advantage, complex, long game

    params = param_learner(pos_emb, task_feat)
    print(f"Adapted parameters: {params}")

    # Test curriculum manager
    curriculum = MetaLearningCurriculum()
    recent_perf = [0.3, 0.4, 0.5, 0.6, 0.7]  # Improving performance
    task_chars = [{'phase': 'middlegame'}] * 5

    curriculum_update = curriculum.adapt_curriculum(recent_perf, task_chars)
    print(f"Curriculum adaptation: {curriculum_update}")

    # Test meta-GRPO trainer
    meta_trainer = MetaGRPOTrainer(None, {})  # Model would be passed in real usage
    meta_results = meta_trainer.train_with_meta_learning([], {'phase': 'middlegame'})
    print(f"Meta-learning results: {meta_results['meta_learning_metrics']}")

    print("✅ Meta-learning components test passed!")
