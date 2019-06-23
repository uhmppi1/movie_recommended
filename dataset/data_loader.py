import numpy as np
import pandas as pd


class DataLoader():
    '''
    dataset : MovieLens Small Datasets
    Small: 100,000 ratings and 3,600 tag applications applied to 9,000 movies by 600 users. Last updated 9/2018.
    https://grouplens.org/datasets/movielens/latest/
    '''

    def __init__(self):
        self.user2index = {}
        self.index2user = {}
        self.movie2index = {}
        self.index2movie = {}



    def num_user(self):
        return len(self.user2index)

    def num_movie(self):
        return len(self.movie2index)

    def get_movie_info(self, movie_list):
        return self.df_movies.loc[movie_list]

    def load_data(self):
        data_path = 'dataset/ml-latest-small/ratings.csv'
        df_ratings = pd.read_csv(
            data_path,
            usecols=['userId', 'movieId', 'rating'],
            dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'})

        movie_path = 'dataset/ml-latest-small/movies.csv'
        self.df_movies = pd.read_csv(movie_path, index_col='movieId')


        unique_users = df_ratings.userId.unique()
        unique_movies = df_ratings.movieId.unique()

        self.user2index = {user:index for index, user in enumerate(unique_users)}
        self.index2user = {index:user for user, index in self.user2index.items()}
        self.movie2index = {movie: index for index, movie in enumerate(unique_movies)}
        self.index2movie = {index: movie for movie, index in self.movie2index.items()}

        df_ratings['user_index'] = df_ratings['userId'].map(lambda x: self.user2index[x])
        df_ratings['movie_index'] = df_ratings['movieId'].map(lambda x: self.movie2index[x])

        # split into train/test set
        msk = np.random.rand(len(df_ratings)) < 0.8
        df_train = df_ratings[msk]
        df_test = df_ratings[~msk]

        X1_train = df_train['user_index'].values    # array([  0,   0,   0, ..., 609, 609, 609], dtype=int64)
        X2_train = df_train['movie_index'].values
        y_train = df_train['rating'].values

        X1_test = df_test['user_index'].values  # array([  0,   0,   0, ..., 609, 609, 609], dtype=int64)
        X2_test = df_test['movie_index'].values
        y_test = df_test['rating'].values

        return (X1_train, X2_train, y_train), (X1_test, X2_test, y_test)
