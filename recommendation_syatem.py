import pandas as pd
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Load MovieLens 100k dataset
data = Dataset.load_builtin('ml-100k')
reader = Reader(rating_scale=(1, 5))

# Convert dataset to Surprise format
trainset, testset = train_test_split(data, test_size=0.25, random_state=42)

# Initialize SVD model
model = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=42)

# Train the model
model.fit(trainset)

# Evaluate the model
predictions = model.test(testset)
rmse = accuracy.rmse(predictions, verbose=False)
mae = accuracy.mae(predictions, verbose=False)

print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")

# Function to get top-N recommendations for a user
def get_top_n_recommendations(model, trainset, user_id, n=5):
    # Get all items the user hasn't rated
    all_items = set(range(trainset.n_items))
    rated_items = set(iid for (iid, _) in trainset.ur[trainset.to_inner_uid(str(user_id))])
    unrated_items = all_items - rated_items
    
    # Predict ratings for unrated items
    predictions = [(item, model.predict(str(user_id), str(item)).est) for item in unrated_items]
    # Sort by predicted rating
    predictions.sort(key=lambda x: x[1], reverse=True)
    
    # Get movie names (load movie metadata)
    movie_data = pd.read_csv('http://files.grouplens.org/datasets/movielens/ml-100k/u.item', sep='|', 
                             encoding='latin-1', header=None, 
                             names=['movie_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL', 
                                    'unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 
                                    'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 
                                    'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'])
    
    # Map movie IDs to titles
    top_n = [(movie_data[movie_data['movie_id'] == int(item)]['title'].iloc[0], rating) 
             for item, rating in predictions[:n]]
    return top_n

# Example: Get top-5 recommendations for user ID 1
user_id = 1
top_n = get_top_n_recommendations(model, trainset, user_id, n=5)

print(f"\nTop-5 recommendations for user {user_id}:")
for i, (movie_title, predicted_rating) in enumerate(top_n, 1):
    print(f"{i}. {movie_title}: {predicted_rating:.2f}")

# Example: Predict rating for a specific user-movie pair
sample_user = 1
sample_movie = 50  # Movie ID for "Star Wars (1977)"
pred = model.predict(str(sample_user), str(sample_movie))
print(f"\nPredicted rating for user {sample_user} on movie 'Star Wars (1977)': {pred.est:.2f}")
