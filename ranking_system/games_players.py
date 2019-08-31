from dataclasses import dataclass
from typing import Dict, List
import pandas as pd
import numpy as np

from messages import GameToSkill, MarginalSkills, SkillToGame, GameToPerformance, PerformanceToGame, \
    MarginalPerformance, Skill


class Player:
    def __init__(self, idx: int, name):
        self.idx = idx
        self.name = name
        self.prior_skill = Skill(mean=0., precision=1.)  # Prior
        self.marginal_skill = MarginalSkills()  # Posterior to be updated
        self.games_played = []

    def update_marginal(self):
        downwards_messages = [
            game.downwards_winner if game.winner_idx == self.idx else game.downwards_losser
            for game in self.games_played
        ]
        self.marginal_skill.update(self.prior_skill, downwards_messages)


class Game:
    def __init__(self, winner_idx: int, losser_idx: int):
        self.winner_idx = winner_idx
        self.losser_idx = losser_idx
        # Messages P1 - upwards / downwards
        self.upwards_winner = SkillToGame()
        self.downwards_winner = GameToSkill(mean=0., precision=0.)
        # Messages P2 - upwards / downwards
        self.upwards_losser = SkillToGame()
        self.downwards_losser = GameToSkill(mean=0., precision=0.)
        # Messages game
        self.upwards_game = GameToPerformance()
        self.downwards_game = PerformanceToGame()
        self.marginal_performance = MarginalPerformance()

    def update_upwards_player_messages(self, players: 'PlayerList'):
        winner, losser = players[self.winner_idx], players[self.losser_idx]
        self.upwards_winner.update(marginal_skill=winner.marginal_skill,
                                   message=self.downwards_winner)
        self.upwards_losser.update(marginal_skill=losser.marginal_skill,
                                   message=self.downwards_losser)

    def update_upwards_game_message(self):
        self.upwards_game.update(winner_message=self.upwards_winner,
                                 losser_message=self.upwards_losser)

    def update_marginal_performance(self):
        self.marginal_performance.update(players_message=self.upwards_game)

    def update_downwards_game_message(self):
        self.downwards_game.update(marginal_performance=self.marginal_performance,
                                   message=self.upwards_game)

    def update_downwards_players_message(self):
        self.downwards_winner.update(top_message=self.downwards_game,
                                     bottom_message=self.upwards_losser,
                                     is_winner=True
                                     )
        self.downwards_losser.update(top_message=self.downwards_game,
                                     bottom_message=self.upwards_winner,
                                     is_winner=False)

    def update_messages(self, players: 'PlayerList'):
        self.update_upwards_player_messages(players)
        self.update_upwards_game_message()
        self.update_marginal_performance()
        self.update_downwards_game_message()
        self.update_downwards_players_message()


@dataclass
class PlayerList:
    player_list: List[Player]

    def get_player_games(self, games: 'GameList'):
        for game in games:
            self.player_list[game.winner_idx].games_played.append(game)
            self.player_list[game.losser_idx].games_played.append(game)

    def update_marginal_skills(self):
        for player in self.player_list:
            player.update_marginal()

    def __getitem__(self, item):
        return self.player_list[item]

    def __len__(self):
        return len(self.player_list)

    @property
    def skills(self) -> np.ndarray:
        N = len(self)
        skills = np.zeros((N, ), dtype=np.float32)
        for id in range(N):
            skills[id] = self[id].marginal_skill.mean
        return skills

    @property
    def precisions(self):
        N = len(self)
        precisions = np.zeros((N, ), dtype=np.float32)
        for id in range(N):
            precisions[id] = self[id].marginal_skill.precision
        return precisions
    @property
    def names(self) -> List:
        return [player.name for player in self.player_list]

    @staticmethod
    def create_from_csv(path, sep=','):
        player_df = pd.read_csv(path, sep=sep)
        player_list = [
            Player(idx=idx, name=name) for idx, name in enumerate(player_df['name'].values)
        ]
        return PlayerList(player_list)


@dataclass
class GameList:
    game_list: List[Game]

    def update_messages(self, players: PlayerList):
        for game in self.game_list:
            game.update_messages(players)

    def __getitem__(self, item):
        return self.game_list[item]

    def __len__(self):
        return len(self.game_list)

    @staticmethod
    def create_from_csv(path, sep=','):
        game_df = pd.read_csv(path, sep=sep)
        game_list = [Game(winner_idx - 1, losser_idx - 1) for winner_idx, losser_idx
                     in game_df[['winner', 'losser']].values
                     ]
        return GameList(game_list)
