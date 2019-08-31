from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from typing import List

from utils import GaussianDistribution, StandardGaussian, Parameter


@dataclass
class Message(GaussianDistribution):
    step: int = 0

    @abstractmethod
    def update(self, *args):
        pass


@dataclass
class Skill(GaussianDistribution):
    step: int = 0

    @abstractmethod
    def update(self, *args):
        pass


@dataclass
class GameToSkill(Message):
    """
    Downwards player message. Last to update
    """

    def update(self,
               top_message: 'PerformanceToGame',
               bottom_message: 'SkillToGame',
               is_winner: bool):
        """
        For a given game and player, top_message represents the performace to game message comming
        from game results, where as bottom_message the skill to game message coming from players.

        Type of update: Marginalising product of Gaussians
        Notes: has to be computed for each game
        """
        assert top_message.step == bottom_message.step
        if is_winner:
            update_mean = bottom_message.mean + top_message.mean
        else:
            update_mean = bottom_message.mean - top_message.mean
        update_variance = 1. + top_message.variance + bottom_message.variance
        self.update_mean_and_precision(updated_mean=update_mean, updated_var=update_variance)
        self.step += 1


@dataclass
class MarginalSkills(Skill):
    """
    EP objective. Updated once the messages from games are updated (i.e: outside games for loop)
    """

    def update(self,
               prior: Skill,
               messages: List[GameToSkill]):
        """
        Priors represent the collection of skill prior distributions (Gaussian) of a player.
        Messages represent a list of downwards player messages from each played game by the player
        Hence, if player_i played G_i games, len(messages)= G_i

        Type of update: Product of gaussians
        """

        update_precision = np.sum(np.array([message.precision for message in messages]))
        update_precision += prior.precision

        update_natural_mean = np.sum(np.array([message.natural_mean for message in messages]))
        update_natural_mean += prior.natural_mean

        self.update_mean_and_precision(updated_pre=update_precision,
                                       updated_nat_mean=update_natural_mean)
        self.step = messages[0].step


@dataclass
class SkillToGame(Message):
    """
    Upwards player message. First to update.
    """

    def update(self,
               marginal_skill: MarginalSkills,
               message: GameToSkill):
        """
        Marginal Skill represent approximate posterior of a player's skill.
        Message represents downwards message to the player from the game

        Type of update: Division of gaussians
        Notes: has to be computed for each game
        """
        assert marginal_skill.step == message.step
        update_precision = marginal_skill.precision - message.precision
        update_natural_mean = marginal_skill.natural_mean - message.natural_mean

        self.update_mean_and_precision(updated_pre=update_precision,
                                       updated_nat_mean=update_natural_mean)
        self.step += 1


@dataclass
class GameToPerformance(Message):
    """
    Upwards game message. Second to update.
    """

    def update(self,
               winner_message: SkillToGame,
               losser_message: SkillToGame):
        """
        For a given game, winner_message and losser_messages are the `SkillToGames` of each
        players the winner and the losser.

        Type of update: Marginalising product of Gaussians
        Notes: has to be computed for each game
        """
        assert winner_message.step == losser_message.step
        update_mean = winner_message.mean - losser_message.mean
        update_variance = 1. + winner_message.variance + losser_message.variance

        self.update_mean_and_precision(updated_mean=update_mean, updated_var=update_variance)
        self.step += 1


@dataclass
class MarginalPerformance(GaussianDistribution):
    """
    Posterior performance approximation through moment matching. Uses upwards game messages.
    Third to update.
    """
    step: int = 0

    def update(self, players_message: GameToPerformance):
        """
        Type of update: MomentMatching
        Notes: has to be computed for each game
        """
        return self._moment_matching_update(players_message)

    def _moment_matching_update(self, players_message: GameToPerformance):
        """
        Explanation of moment matching
        """
        normal = StandardGaussian()

        update_mean = players_message.mean + players_message.std * normal.psi(
            players_message.mean / players_message.std
        )
        update_variance = players_message.variance * (1. - normal.lambda_(
            players_message.mean / players_message.std
        ))

        self.update_mean_and_precision(updated_mean=update_mean, updated_var=update_variance)
        self.step += 1


@dataclass
class PerformanceToGame(Message):
    """
    Downwards game message. Second to last to update.
    """

    def update(self,
               marginal_performance: MarginalPerformance,
               message: GameToPerformance):
        """
        For a given game, marginal performances performance approximate posterior distributions
        (Gaussian). Message represents an incoming game message from the game to the performance.

        Type of update: Division of gaussians
        Notes: has to be computed for each game
        """
        assert marginal_performance.step == message.step
        update_precision = marginal_performance.precision - message.precision
        update_natural_mean = marginal_performance.natural_mean - message.natural_mean
        self.update_mean_and_precision(updated_pre=update_precision,
                                       updated_nat_mean=update_natural_mean)
        self.step += 1


