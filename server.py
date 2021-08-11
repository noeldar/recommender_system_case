from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
import joblib
import datetime as dt

from sklearn.metrics.pairwise import cosine_similarity
import tensorflow
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.layers import Input, Dense, Dropout, GRU

app = Flask(__name__)

def item_indices(data, session_key='sessionid', item_key='productid', time_key='time',
                 itemmap=None):
    data.sort_values([session_key, time_key], inplace=True)
    """
            Add item index column named "item_idx" to the df
            Args:
                itemmap (pd.DataFrame): mapping between the item Ids and indices
    """
    if itemmap is None:
        item_ids = data[item_key].unique()  # unique item ids
        item2idx = pd.Series(data=np.arange(len(item_ids)),
                             index=item_ids)
        itemmap = pd.DataFrame({item_key: item_ids,
                                'item_idx': item2idx[item_ids].values})

    itemmap = itemmap
    data = pd.merge(data, itemmap, on=item_key, how='inner')
    return data,itemmap

#loading model, previously added items, product features and meta information of the items
model1 = tensorflow.keras.models.load_model("model/GRU4REC_9.h5")
train_data = pd.read_csv('model/train_tr.txt', sep='\t', dtype={'productid': str})
train_n_items = len(train_data['productid'].unique()) + 1
train_dataset, itemmap = item_indices(train_data)

meta_df_reorganized=pd.read_csv('model/meta_df_reorganized.csv',index_col=0)
item_features = pd.read_csv('model/item_features.txt', sep='\t',index_col=0)

# calculating cosine similarity matrix with product features
sg = cosine_similarity(item_features.values, item_features.values)
item_similarity=pd.DataFrame(sg, index=item_features.index, columns=item_features.index)
top_N = 10


@app.route("/predict", methods=['POST'])
def do_prediction():
    #getting the session json to recommend the next 10 item
    json = request.get_json()

    test_data = pd.DataFrame(json, index=[0])

    recommends_forsimilars={}
    recommends_forall={}

    # getting current bag of each session
    test_dataset_forSimilarity = test_data.groupby('sessionid')['productid'].apply(list).reset_index()

    for index, row in test_dataset_forSimilarity.iterrows():
        #for a session getting current products' similarities with other products from similarity matrix
        partofsg = item_similarity.loc[row['productid']]

        #getting column indexes of top top_N + partofsg.shape[0](since there will be 1.) similarity scores
        top = top_N + partofsg.shape[0]
        idx = np.argpartition(partofsg.values, partofsg.values.size - top, axis=None)[-top:]
        result = np.column_stack(np.unravel_index(idx, partofsg.values.shape))
        indices = result[:, 1]

        #getting the product ids of most similar(recommended) products
        recommends = list(set([item_similarity.columns[i] for i in indices]))

        #deleting products that are in the current bag
        for product in row['productid']:
            recommends.remove(product)

        #getting recommended items' name
        similar_recommends = meta_df_reorganized[(meta_df_reorganized["productid"].isin(recommends))]['name'].values
        recommends_forsimilars[row['sessionid']] = ",".join(similar_recommends)

    test_data['time'] = test_data['eventtime'].apply(
        lambda x: dt.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%fZ').timestamp())
    test_data = test_data.drop(columns="eventtime")

    # check if test data contains any element that is not in RNN's itemmapper
    fark = list(set(test_data['productid'].values) - set(itemmap['productid'].values))

    # if est data doesn't contain any element that is not in RNN's itemmapper
    # we will make recommendations with session-based and content-based recommender

    if len(fark) == 0:
        # justifying product ids, getting the last product of every session in postted json
        test_dataset, _ = item_indices(test_data, itemmap=itemmap)
        idx = test_dataset.groupby(['sessionid'])['time'].transform(max) == test_dataset['time']
        df = test_dataset[idx]

        feat = df['item_idx'].values

        # adjusting the input data's shape to make prediction
        if len(feat) % 512 > 0:
            mod = len(feat) % 512
            toBeAdd = 512 - mod
            X = np.zeros((toBeAdd, train_n_items))
            input_oh = to_categorical(feat, num_classes=train_n_items)
            input_oh = np.vstack((input_oh, X))
            input_oh = np.expand_dims(input_oh, axis=1)

        # making prediction
        pred = model1.predict(input_oh, batch_size=512)

        for row_idx in range(len(feat)):
            # getting the prediction scores for each last item of all the sessions
            pred_row = pred[row_idx]
            # getting the top 10 prediction scores' ids
            rec_idx = pred_row.argsort()[-5:][::-1]

            # with these ids, getting the names of the products from meta.json

            recommends = ""
            for i in range(len(rec_idx)):
                temp = itemmap[(itemmap["item_idx"] == rec_idx[i])]['productid'].values[0]
                recommends = recommends + "," + \
                             meta_df_reorganized[(meta_df_reorganized["productid"] == temp)]['name'].values[0]

            # getting content-based recommendations
            similar_ones=recommends_forsimilars[df['sessionid'].values[row_idx]]
            similars=",".join(similar_ones.split(",")[0:5])


            recommends_forall[df['sessionid'].values[row_idx]] = recommends+ similars

    else:
        recommends_forall = recommends_forsimilars









    return jsonify(recommends_forall)



if __name__ == "__main__":
    app.run(host='0.0.0.0')
