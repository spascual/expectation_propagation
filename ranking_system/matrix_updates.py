from typing import Tuple

import numpy as np
import scipy.stats

from matrix_stats import GameStats, PlayerStats, WinLossHistory


class MatrixMessageUpdates:

    @staticmethod
    def update_upwards_player_messages(game_stats: GameStats) -> GameStats:
        marginal_w_precision = game_stats[:, 7, 1]
        marginal_w_natural_mean = _natural_mean(marginal_w_precision, game_stats[:, 7, 0])
        downwards_w_precision = game_stats[:, 6, 1]
        downwards_w_natural_mean = _natural_mean(downwards_w_precision, game_stats[:, 6, 0])
        # Winner update
        game_stats[:, 0, 1] = marginal_w_precision - downwards_w_precision
        if np.min(game_stats[:, 0, 1], axis=0) < 0:
            raise ValueError
        upwards_w_natural_mean = marginal_w_natural_mean - downwards_w_natural_mean
        game_stats[:, 0, 0] = _mean(game_stats[:, 0, 1], upwards_w_natural_mean)

        marginal_l_precision = game_stats[:, 8, 1]
        marginal_l_natural_mean = _natural_mean(marginal_l_precision, game_stats[:, 8, 0])
        downwards_l_precision = game_stats[:, 5, 1]
        downwards_l_natural_mean = _natural_mean(downwards_l_precision, game_stats[:, 5, 0])
        # Loser update
        game_stats[:, 1, 1] = marginal_l_precision - downwards_l_precision
        if np.min(game_stats[:, 1, 1], axis=0) < 0:
            raise ValueError
        upwards_l_natural_mean = marginal_l_natural_mean - downwards_l_natural_mean
        game_stats[:, 1, 0] = _mean(game_stats[:, 1, 1], upwards_l_natural_mean)
        return game_stats

    @staticmethod
    def update_upwards_game_message(game_stats: GameStats) -> GameStats:
        upwards_w_mean, upwards_w_precision = game_stats[:, 0, 0], game_stats[:, 0, 1]
        upwards_l_mean, upwards_l_precision = game_stats[:, 1, 0], game_stats[:, 1, 1]
        # update
        game_stats[:, 2, 0] = upwards_w_mean - upwards_l_mean
        game_stats[:, 2, 1] = 1. / (1. + 1. / upwards_w_precision + 1. / upwards_l_precision)
        return game_stats

    @staticmethod
    def update_marginal_performance(game_stats: GameStats) -> GameStats:
        """
        Updates suffient statistics of an approximation of marginal performance using moment
        matching.
        """
        upwards_game_mean, upwards_game_precision = game_stats[:, 2, 0], game_stats[:, 2, 1]
        upwards_game_var = _var(upwards_game_precision)
        upwards_game_std = _std(upwards_game_precision)
        # Moment matching update
        game_stats[:, 3, 0] = upwards_game_mean + upwards_game_std * _psi(
            upwards_game_mean / upwards_game_std
        )
        game_stats[:, 3, 1] = _precision(upwards_game_var * (1. - _lambda(
            upwards_game_mean / upwards_game_std
        )))

        return game_stats

    @staticmethod
    def update_downwards_game_message(game_stats: GameStats) -> GameStats:
        performance_precision = game_stats[:, 3, 1]
        performance_natural_mean = _natural_mean(performance_precision, game_stats[:, 3, 0])
        upwards_game_precision = game_stats[:, 2, 1]
        upwards_game_natural_mean = _natural_mean(upwards_game_precision, game_stats[:, 2, 0])
        # update
        game_stats[:, 4, 1] = performance_precision - upwards_game_precision
        downwards_game_natural_mean = performance_natural_mean - upwards_game_natural_mean
        game_stats[:, 4, 0] = _mean(game_stats[:, 4, 1], downwards_game_natural_mean)
        return game_stats

    @staticmethod
    def update_downwards_players_message(game_stats: GameStats) -> GameStats:
        downwards_game_mean, downwards_game_precision = game_stats[:, 4, 0], game_stats[:, 4, 1]
        # Winner update
        upwards_l_mean, upwards_l_precision = game_stats[:, 1, 0], game_stats[:, 1, 1]
        game_stats[:, 6, 0] = upwards_l_mean + downwards_game_mean
        game_stats[:, 6, 1] = 1. / (1. + 1. / upwards_l_precision + 1. / downwards_game_precision)

        # losser update
        upwards_w_mean, upwards_w_precision = game_stats[:, 0, 0], game_stats[:, 0, 1]
        game_stats[:, 5, 0] = upwards_w_mean - downwards_game_mean
        game_stats[:, 5, 1] = 1. / (1. + 1. / upwards_w_precision + 1. / downwards_game_precision)
        return game_stats

    def update_game_messages(self, game_stats: GameStats) -> GameStats:
        game_stats = self.update_upwards_player_messages(game_stats)
        game_stats = self.update_upwards_game_message(game_stats)
        game_stats = self.update_marginal_performance(game_stats)
        game_stats = self.update_downwards_game_message(game_stats)
        game_stats = self.update_downwards_players_message(game_stats)
        return game_stats


class MatrixSkillUpdates:

    @staticmethod
    def compute_message_games(game_stats: GameStats,
                              player_stats: PlayerStats,
                              history: WinLossHistory):
        downwards_l_precision = game_stats[:, 5, 1]
        downwards_l_natural_mean = _natural_mean(downwards_l_precision, game_stats[:, 5, 0])
        downwards_w_precision = game_stats[:, 6, 1]
        downwards_w_natural_mean = _natural_mean(downwards_w_precision, game_stats[:, 6, 0])
        # Updates

        for player_id in history.player_ids:
            loss_idx, win_idx = history.losses[player_id], history.wins[player_id]
            player_stats[player_id, 2, 1] = np.sum(downwards_l_precision[loss_idx])
            player_stats[player_id, 2, 1] += np.sum(downwards_w_precision[win_idx])
            player_stats[player_id, 2, 0] = np.sum(downwards_l_natural_mean[loss_idx])
            player_stats[player_id, 2, 0] += np.sum(downwards_w_natural_mean[win_idx])
        return player_stats

    @staticmethod
    def update_marginal(player_stats: PlayerStats) -> PlayerStats:
        messages_precision, messages_natural_mean = player_stats[:, 2, 1], player_stats[:, 2, 0]
        prior_precision = player_stats[:, 1, 1]
        prior_natural_mean = _natural_mean(prior_precision, player_stats[:, 1, 0])
        # Update
        player_stats[:, 0, 1] = prior_precision + messages_precision
        marginal_natural_mean = prior_natural_mean + messages_natural_mean
        player_stats[:, 0, 0] = _mean(player_stats[:, 0, 1], marginal_natural_mean)
        return player_stats

    @staticmethod
    def insert_marginal_into_games(game_stats: GameStats,
                                   player_stats: PlayerStats,
                                   history: WinLossHistory) -> GameStats:
        marginal_mean, marginal_precision = player_stats[:, 0, 0], player_stats[:, 0, 1]
        for player_id in history.player_ids:
            loss_idx, win_idx = history.losses[player_id], history.wins[player_id]
            game_stats[win_idx, 7, 0] = np.tile(marginal_mean[player_id - 1], (len(win_idx),))
            game_stats[win_idx, 7, 1] = np.tile(marginal_precision[player_id - 1], (len(win_idx),))
            game_stats[loss_idx, 8, 0] = np.tile(marginal_mean[player_id - 1], (len(loss_idx),))
            game_stats[loss_idx, 8, 1] = np.tile(marginal_precision[player_id - 1],
                                                 (len(loss_idx),))
        return game_stats

    def update_skill_marginals(self,
                               game_stats: GameStats,
                               player_stats: PlayerStats,
                               history: WinLossHistory) -> Tuple[GameStats, PlayerStats]:
        player_stats = self.compute_message_games(game_stats, player_stats, history)
        player_stats = self.update_marginal(player_stats)
        game_stats = self.insert_marginal_into_games(game_stats, player_stats, history)
        return game_stats, player_stats


def _natural_mean(precision, mean):
    return precision * mean


def _mean(precision, natural_mean):
    return natural_mean / precision


def _var(precision):
    assert np.min(precision, axis=0) > 0.
    return 1. / precision


def _precision(var):
    return 1. / var


def _std(precision):
    return np.sqrt(1. / precision)


def _psi(x: np.ndarray) -> np.ndarray:
    return scipy.stats.norm.pdf(x) / scipy.stats.norm.cdf(x)


def _lambda(x: np.ndarray) -> np.ndarray:
    return _psi(x) * (_psi(x) + x)
