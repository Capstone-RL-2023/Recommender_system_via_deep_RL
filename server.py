import pandas as pd
import numpy as np
import os
import json

from envs import OfflineEnv
from recommender import DRRAgent
from eval import evaluate
from flask import Flask, jsonify

ROOT_DIR = os.getcwd()
DATA_DIR = os.path.join(ROOT_DIR, 'ml-1m')
SAVED_ACTOR = './save_weight/actor_model.h5'
SAVED_CRITIC = './save_weight/critic_model.h5'
STATE_SIZE = 10
app = Flask(__name__)

@app.route('/recommend', methods=['GET'])
def get_recommendations():
    user_id = requrest.args.get('user_id')
    response = {"success": False,
                "user_id": user_id,
                "recommendations": recommendations}
    
    # Load and modify dataset
    try:
        users_dict = np.load("./data/user_dict.npy", )
        users_history_lens = np.load("./data/users_histroy_len.npy", )
        ratings_list = [i.strip().split("::") for i in open(os.path.join(DATA_DIR,'ratings.dat'), 'r').readlines()]
        users_list = [i.strip().split("::") for i in open(os.path.join(DATA_DIR,'users.dat'), 'r').readlines()]
        movies_list = [i.strip().split("::") for i in open(os.path.join(DATA_DIR,'movies.dat'),encoding='latin-1').readlines()]
        ratings_df = pd.DataFrame(ratings_list, columns = ['UserID', 'MovieID', 'Rating', 'Timestamp'], dtype = np.uint32)
        movies_id_to_movies = {movie[0]: movie[1:] for movie in movies_list}
        
        ratings_df = ratings_df.applymap(int)
        ratings_df = ratings_df.sort_values(by='Timestamp', ascending=True)
        
        # Set parameters
        users_num = max(ratings_df["UserID"])+1
        items_num = max(ratings_df["MovieID"])+1
        eval_users_num = int(users_num * 0.2)
        eval_users_dict = {k:users_dict[k] for k in range(users_num-eval_users_num, users_num)}
        
        # Get recommendations
        env = OfflineEnv(eval_users_dict, users_history_lens, movies_id_to_movies, STATE_SIZE, fix_user_id=user_id)
        recommender = DRRAgent(env, users_num, items_num, STATE_SIZE)
        recommender.actor.build_networks()
        recommender.critic.build_networks()
        recommender.load_model(saved_actor, saved_critic)
        precision, ndcg = evaluate(recommender, env, check_movies=True, top_k=TOP_K) # if check movies is true, you can check the recommended movies
    except Exception as e:
        response["error"] = str(e)
     
    return response, 200
    
if __name__ == '__main__':
    app.run(debug=True)