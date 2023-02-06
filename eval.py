import numpy as np

# Version from Shanu


def evaluate(recommender, env, check_movies=False, top_k=False):
    # episodic reward
    mean_precision = 0
    mean_ndcg = 0
    episode_reward = 0
    steps = 0
    # Environment    
    user_id, items_ids, done = env.reset()
    while not done:
        #print("user_id :",user_id)        
        # Observe current state & Find action        
        # Embedding        
        user_eb = recommender.embedding_network.get_layer(
            'user_embedding')(np.array(user_id))
        items_eb = recommender.embedding_network.get_layer(
            'movie_embedding')(np.array(items_ids))
        # SRM state        
        state = recommender.srm_ave(
            [np.expand_dims(user_eb, axis=0), np.expand_dims(items_eb, axis=0)])
        # Action(ranking score)        
        action = recommender.actor.network(state)
        # Item        
        recommended_item = recommender.recommend_item(
            action, env.recommended_items, top_k=top_k)
        if check_movies:
            print(f'recommended items ids : {recommended_item}')
            print(
                f'recommened items : \n {np.array(env.get_items_names(recommended_item), dtype=object)}')
        next_items_ids, reward, done, _ = env.step(
            recommended_item, top_k=top_k)
        print("done :", done)

        if top_k:
            correct_list = [1 if r > 0 else 0 for r in reward]
            # ndcg
            dcg, idcg = calculate_ndcg(
                correct_list, [1 for _ in range(len(reward))])
            mean_ndcg += dcg/idcg

            # precision
            correct_num = top_k-correct_list.count(0)
            mean_precision += correct_num/top_k

            # precision
        reward = np.sum(reward)
        items_ids = next_items_ids
        episode_reward += reward
        steps += 1
        if check_movies:
            print(
                f'precision : {correct_num/top_k}, dcg : {dcg:0.3f}, idcg : {idcg:0.3f}, ndcg : {dcg/idcg:0.3f}, reward : {reward}')
        break

    if check_movies:
        print(
            f'precision : {mean_precision/steps}, ngcg : {mean_ndcg/steps}, episode_reward : {episode_reward}')
    return np.array(env.get_items_names(recommended_item), dtype=object), mean_precision, mean_ndcg, reward


def calculate_ndcg(rel, irel):
    dcg = 0
    idcg = 0
    rel = [1 if r > 0 else 0 for r in rel]
    for i, (r, ir) in enumerate(zip(rel, irel)):
        dcg += (r)/np.log2(i+2)
        idcg += (ir)/np.log2(i+2)
    return dcg, idcg
