from collections import defaultdict
from dataclasses import dataclass
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd

PlayerStats = np.ndarray
GameStats = np.ndarray


@dataclass
class WinLossHistory:
    wins: Dict
    losses: Dict

    @staticmethod
    def create_from_dataframe(game_df: pd.DataFrame):
        K = len(game_df)
        wins = defaultdict(list)
        losses = defaultdict(list)
        for i in range(K):
            winner_idx, losser_idx = game_df['winner'].loc[i] - 1, game_df['losser'].loc[i] - 1
            wins[winner_idx].append(i)
            losses[losser_idx].append(i)
        return WinLossHistory(wins, losses)

    @property
    def player_ids(self):
        return set(self.wins).union(self.losses)


class PlayerStatsFactory:
    """
    Let N be the number of players
    """

    @staticmethod
    def create_from_csv(path, sep=',') -> Tuple[PlayerStats, List]:
        player_df = pd.read_csv(path, sep=sep)
        player_names = player_df['name'].values
        N = len(player_df)
        player_stats = np.zeros((N, 3, 2), dtype=np.float32)
        player_stats[:, 1, 0] = np.zeros((N,), dtype=np.float32)
        player_stats[:, 1, 1] = np.ones((N,), dtype=np.float32)
        return player_stats, player_names


class GameStatsFactory:
    """
    Let K be the number of Games
    """

    @staticmethod
    def create_from_csv(path, sep=',') -> Tuple[GameStats, WinLossHistory]:
        game_df = pd.read_csv(path, sep=sep)
        K = len(game_df)
        history = WinLossHistory.create_from_dataframe(game_df)
        game_stats = np.zeros((K, 9, 2), dtype=np.float32)
        game_stats[:, -2:, 0] = np.zeros((K, 2,), dtype=np.float32)
        game_stats[:, -2:, 1] = np.zeros((K, 2,), dtype=np.float32)
        return game_stats, history
