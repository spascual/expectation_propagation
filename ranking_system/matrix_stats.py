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
    N: int

    @staticmethod
    def create_from_dataframe(game_df: pd.DataFrame, N: int):
        K = len(game_df)
        wins = defaultdict(list)
        losses = defaultdict(list)
        for i in range(K):
            winner_idx = int(game_df['winner'].loc[i]) - 1
            losser_idx = int(game_df['losser'].loc[i]) - 1
            wins[winner_idx].append(i)
            losses[losser_idx].append(i)
        return WinLossHistory(wins, losses, N)

    @property
    def player_ids(self):
        return list(range(self.N))


class PlayerStatsFactory:
    """
    Let N be the number of players
    """

    @staticmethod
    def create_from_csv(path, sep=',', name_col='name') -> Tuple[PlayerStats, List, int]:
        player_df = pd.read_csv(path, sep=sep)
        player_names = list(set(player_df[name_col].values))
        N = len(player_names)
        player_stats = np.zeros((N, 3, 2), dtype=np.float32)
        player_stats[:, 1, 0] = np.zeros((N,), dtype=np.float32)
        player_stats[:, 1, 1] = np.ones((N,), dtype=np.float32)
        return player_stats, player_names, N

    @staticmethod
    def create_from_dataframe(player_df: pd.DataFrame, name_col='name'):
        player_names = list(set(player_df[name_col].values))
        N = len(player_names)
        player_stats = np.zeros((N, 3, 2), dtype=np.float32)
        player_stats[:, 1, 0] = np.zeros((N,), dtype=np.float32)
        player_stats[:, 1, 1] = np.ones((N,), dtype=np.float32)
        return player_stats, player_names, N


class GameStatsFactory:
    """
    Let K be the number of Games
    """

    @staticmethod
    def create_from_csv(path, N: int, sep=',') -> Tuple[GameStats, WinLossHistory]:
        game_df = pd.read_csv(path, sep=sep)
        K = len(game_df)
        history = WinLossHistory.create_from_dataframe(game_df, N)
        game_stats = np.zeros((K, 9, 2), dtype=np.float32)
        game_stats[:, -2:, 0] = np.zeros((K, 2,), dtype=np.float32)
        game_stats[:, -2:, 1] = np.zeros((K, 2,), dtype=np.float32)
        return game_stats, history

    @staticmethod
    def create_from_dataframe(game_df: pd.DataFrame, N: int) -> Tuple[GameStats, WinLossHistory]:
        K = len(game_df)
        history = WinLossHistory.create_from_dataframe(game_df, N)
        game_stats = np.zeros((K, 9, 2), dtype=np.float32)
        game_stats[:, -2:, 0] = np.zeros((K, 2,), dtype=np.float32)
        game_stats[:, -2:, 1] = np.zeros((K, 2,), dtype=np.float32)
        return game_stats, history
