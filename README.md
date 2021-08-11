# recommender_system_case

Run the comments below for the API. 
```
docker build -t ml-model .
docker run -d -p 5000:5000 ml-model
curl --location --request POST 'http://localhost:5000/predict' --header 'Content-Type: application/json' --data-raw '{"sessionid": "14e205fd-73d7-4eff-9327-708689bdea33","productid":"ZYHPDROETYRD010","eventtime": "2020-06-01T08:59:46.580Z"}'
```

Description of the scripts in the project: 

- **Overall_preprocessing/preprocessing.ipynb** : Deletion of instances containing None, handling of illegal situations (negatively priced products, whether a product is in event data but not in metadata, etc.), cleaning of sessions and products that are less active and popular than a certain threshold 
- **session-based/preprocess/TrainTestSplit_sessionbased.ipynb** : Test and train split for the GRU model used in the session based recommender. The points that are considered while making this split have been added as comments in the script. 
- **session-based/gru4rec.py** : Training and evaluating GRU model for session-based recommender
- **content-based/SimilarityfromWord2Vec.ipynb** : Extracting each products' features by using bag of words and the,r word2vec embeddings. After each product's features were determined, cosine similarity matrix calculated. Cosine similarity matrix was used for content-based recommender
- **server.py** : Flask API codes for deploying content-based and session-based recommenders.