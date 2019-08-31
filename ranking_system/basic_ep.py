import time

import numpy as np

from games_players import PlayerList, GameList
from utils import StandardGaussian


class BasicEPLoop:
    """
    Basic EP loop executing belief progation from games outcomes to player skill through message
    passing.
    """

    def __init__(self, players: PlayerList, games: GameList):
        self.players = players
        self.games = games
        self.players.get_player_games(games)

    def run(self, num_iterations, logging=True):
        t0 = time.time()
        for tau in range(num_iterations):
            self.players.update_marginal_skills()
            self.games.update_messages(self.players)
            if logging and (tau + 1) % (num_iterations // 4) == 0:
                print(f'#### EP iteration #{tau + 1} completed #### time elapsed {time.time()-t0}')
        self.players.update_marginal_skills()

    # TODO
    def simulate_game(self, player_1: int, player_2: int, logging=True):
        p1_mean, p2_mean = self.players.skills[player_1], self.players.skills[player_2]
        p1_prec, p2_prec = self.players.precisions[player_1], self.players.precisions[player_2]
        probs_p1_wins = StandardGaussian().cdf(
            (p1_mean - p2_mean) / np.sqrt(1. + 1. / p1_prec + 1. / p2_prec))
        did_p1_win = np.random.binomial(1, probs_p1_wins)
        if did_p1_win:
            winner, losser = player_1, player_2
        else:
            winner, losser = player_2, player_1
        if logging:
            print(f'{self.players.names[winner]} beated {self.players.names[losser]}')
        return winner, losser

    def produce_ranking(self):
        """
        Produce a top-10 ranking of the players based on the marginal skill distribution mean.
        """
        skills = self.players.skills
        player_names = self.players.names
        player_names = np.take_along_axis(np.array(player_names), skills.argsort(), axis=0)
        top = 10
        for position, player_name in enumerate(reversed(player_names[-top:])):
            print(f"Position #{position + 1}: {player_name}")


if __name__ == '__main__':
    players = PlayerList.create_from_csv('data/players.csv')
    games = GameList.create_from_csv('data/games.csv')
    ep = BasicEPLoop(players, games)
    ep.run(60)
    ep.produce_ranking()
    ep.simulate_game(0, 1)
