import time
from typing import List

import numpy as np

from matrix_stats import PlayerStats, GameStats, WinLossHistory, GameStatsFactory, \
    PlayerStatsFactory
from matrix_updates import MatrixMessageUpdates, MatrixSkillUpdates
from utils import GaussianDistribution, StandardGaussian


class MatrixEPLoop:
    """
    Basic EP loop executing belief progation from games outcomes to player skill through message
    passing.
    """

    def __init__(self,
                 players: PlayerStats,
                 games: GameStats,
                 history: WinLossHistory,
                 player_names: List):
        self.player_stats = players
        self.game_stats = games
        self.history = history
        self.player_names = player_names

    def run(self, num_iterations, logging=True):
        t0 = time.time()
        for tau in range(num_iterations):
            self.game_stats, self.player_stats = MatrixSkillUpdates().update_skill_marginals(
                self.game_stats, self.player_stats, self.history)
            self.game_stats = MatrixMessageUpdates().update_game_messages(self.game_stats)
            if logging and (tau + 1) % (num_iterations // 2) == 0:
                print(f'#### EP iteration #{tau + 1} completed #### time elapsed {time.time()-t0}')
        self.game_stats, self.player_stats = MatrixSkillUpdates().update_skill_marginals(
            self.game_stats, self.player_stats, self.history)

    def simulate_game(self, player_1: int, player_2: int, logging=True):
        p1_mean, p2_mean = self.player_stats[player_1, 0, 0], self.player_stats[player_2, 0, 0]
        p1_prec, p2_prec = self.player_stats[player_1, 0, 1], self.player_stats[player_2, 0, 1]
        probs_p1_wins = StandardGaussian().cdf(
            (p1_mean - p2_mean) / np.sqrt(1. + 1. / p1_prec + 1. / p2_prec))
        did_p1_win = np.random.binomial(1, probs_p1_wins)
        if did_p1_win:
            winner, losser = player_1, player_2
        else:
            winner, losser = player_2, player_1
        if logging:
            print(f'{self.player_names[winner]} beated {self.player_names[losser]}')
        return winner, losser

    def produce_ranking(self):
        """
        Produce a top-10 ranking of the players based on the marginal skill distribution mean.
        """
        skills = self.player_stats[:, 0, 0]
        player_names = np.take_along_axis(np.array(self.player_names), skills.argsort(), axis=0)
        top = 10
        for position, player_name in enumerate(reversed(player_names[-top:])):
            print(f"Position #{position + 1}: {player_name}")


if __name__ == '__main__':
    players, player_names = PlayerStatsFactory.create_from_csv('data/players.csv')
    games, history = GameStatsFactory.create_from_csv('data/games.csv')
    ep = MatrixEPLoop(players, games, history, player_names)
    ep.run(100)
    ep.produce_ranking()
    ep.simulate_game(0, 1)
