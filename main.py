from dataset.data_loader import DataLoader
from model import generalized_matrix_factorization
import train, infer
import numpy as np

# to reproduce same training result
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

if __name__ == "__main__":
    dataloader = DataLoader()

    (X1_train, X2_train, y_train), (X1_test, X2_test, y_test) = dataloader.load_data()

    print(X1_train.shape)
    print(X2_train.shape)
    print(y_train.shape)
    print(X1_test.shape)
    print(X2_test.shape)
    print(y_test.shape)

    X_train = [X1_train, X2_train]
    X_test = [X1_test, X2_test]

    model, inf_model = generalized_matrix_factorization.get_model(
        dataloader.num_user(), dataloader.num_movie(), latent_dim=10, vu_reg=0.01, vi_reg=0.01)

    model, history = train.train_model(model, X_train, y_train, epochs=10)

    # make prediction using test data
    y_hats = model.predict(X_test)
    y_hats = np.squeeze(y_hats)
    print(y_hats.shape)
    rmse = np.sqrt(sum((y_hats - y_test) ** 2) / len(y_test))
    print(rmse.shape)
    print('Testset RMSE: %.4f' % rmse)

    # get recommendations
    X1_infer = np.expand_dims(np.array([0]), axis=0)  # 0번 유저의 전체 영화에 대한 선호도 조사.
    X2_infer = np.expand_dims(range(dataloader.num_movie()), axis=0)  # 0~9767번 영화
    inf_results = inf_model.predict([X1_infer, X2_infer])
    print(inf_results.shape)
    inf_results = np.squeeze(inf_results)
    print(inf_results.shape)

    recommended_index = inf_results.argsort()[::-1][:10].tolist()
    recommended_movie_ids = [dataloader.index2movie[idx] for idx in recommended_index]
    print(dataloader.get_movie_info(recommended_movie_ids))
