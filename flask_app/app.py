import os

import pandas as pd
from flask import Flask, render_template, url_for, redirect
from pymongo import MongoClient

from ranking_system.matrix_ep import MatrixEPLoop
from ranking_system.matrix_stats import PlayerStatsFactory, GameStatsFactory
from sampling import MatchGenerator

client = MongoClient('mongodb://localhost:27017')
db = client.SentencesDatabase
client.close()
games_db = db['_games']

HOME_DIR = os.environ['HOME']
CELEB_DIR = HOME_DIR + '/celebrity_dataset/'
app = Flask(__name__, static_folder=CELEB_DIR + 'img_align_celeba')


@app.route('/', methods=['GET'])
def home_page():
    next_path = '/stats'
    match_gen = MatchGenerator.create_from_csv(CELEB_DIR + 'clustered_celeb.txt')
    match = match_gen.generate_random_match()
    left_image_path, left_id = match['left'][0], match['left'][1]
    right_image_path, right_id = match['right'][0], match['right'][1]
    num_matches = len(list(games_db.find({}))) + 1

    return render_template('index.html',
                           left_image_path=left_image_path,
                           right_image_path=right_image_path,
                           left_id=left_id,
                           right_id=right_id,
                           next_path=next_path,
                           num_matches=num_matches,
                           ranking_url=url_for('ranking')
                           )


@app.route('/stats/<winner>/<losser>', methods=['GET'])
def stats(winner, losser):
    games_db.insert_one({
        'winner': int(winner) + 1,
        'losser': int(losser) + 1,
    })
    return redirect(url_for('home_page'))


@app.route('/ranking', methods=['GET'])
def ranking():
    game_df = pd.DataFrame(games_db.find({}))
    players_df = pd.read_csv(CELEB_DIR + 'clustered_celeb.txt', sep=',')
    players, player_names, N = PlayerStatsFactory.create_from_dataframe(
        players_df, name_col='group_id')
    games, history = GameStatsFactory.create_from_dataframe(game_df, N)
    ep = MatrixEPLoop(players, games, history, player_names)
    ep.run(100, logging=False)
    first_position = ep.produce_ranking(logging=False)[0]
    return f"{read_group(players_df, first_position)}", 200


def read_group(df, group_id):
    group_first_item = df.loc[df['group_id'] == group_id].iloc[0][df.columns[2:]]
    return str(group_first_item.replace(-1, 'No').replace(1, 'Yes'))[:-40]


if __name__ == '__main__':
    app.run(debug=True)
