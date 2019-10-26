from typing import Dict

import pandas as pd
import numpy as np
import os

HOME_DIR = os.environ['HOME']


class MatchGenerator:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.group_ids = list(set(df['group_id']))
        self.num_groups = len(self.group_ids)

    @staticmethod
    def create_from_csv(path: str):
        df = pd.read_csv(path)
        return MatchGenerator(df)

    def generate_random_match(self) -> Dict:
        left_group, right_group = np.random.choice(self.group_ids, 2, replace=False)
        left_img_path, left_group_id = self.sample_from_group(left_group)
        right_img_path, right_group_id = self.sample_from_group(right_group)
        return dict(left=(left_img_path, left_group_id),
                    right=(right_img_path, right_group_id))

    def sample_from_group(self, group_id):
        sample = self.df.loc[self.df['group_id'] == group_id].sample()
        img_path = sample['id'].to_string().split(' ')[-1]
        return img_path, group_id


if __name__ == '__main__':
    match_gen = MatchGenerator.create_from_csv(HOME_DIR + '/celebrity_dataset/clustered_celeb.txt')
    print(match_gen.generate_random_match())
