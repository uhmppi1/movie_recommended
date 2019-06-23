from keras.layers import Input, Embedding, Flatten, Dense, Multiply
from keras.regularizers import l2
from keras.models import Model


def get_model(num_users, num_items, latent_dim, vu_reg, vi_reg):

    user_input = Input(shape=(1,), dtype='int32', name='user_input')
    item_input = Input(shape=(1,), dtype='int32', name='item_input')
    item_input_for_infer = Input(shape=(num_items,), dtype='int32', name='item_input')

    MF_Embedding_User = Embedding(
        input_dim=num_users,
        output_dim=latent_dim,
        # embeddings_initializer='uniform',
        name='user_embedding',
        embeddings_regularizer=l2(vu_reg))
    MF_Embedding_Item = Embedding(
        input_dim=num_items,
        output_dim=latent_dim,
        # embeddings_initializer='uniform',
        name='item_embedding',
        embeddings_regularizer=l2(vi_reg))

    # Crucial to flatten an embedding vector!
    user_latent = Flatten()(MF_Embedding_User(user_input))
    item_latent = Flatten()(MF_Embedding_Item(item_input))
    item_latent_for_infer = MF_Embedding_Item(item_input_for_infer)

    # Element-wise product of user and item embeddings
    predict_vector = Multiply()([user_latent, item_latent])
    predict_vector_for_infer = Multiply()([user_latent, item_latent_for_infer])

    # Final prediction layer
    prediction = Dense(1, kernel_initializer='glorot_uniform', name='prediction')(predict_vector)
    prediction_for_infer = Dense(1, kernel_initializer='glorot_uniform', name='prediction')(predict_vector_for_infer)

    # Stitch input and output
    model = Model([user_input, item_input], prediction, name='generalized_matrix_factorization')
    print('########## Model for Training ###############')
    model.summary()

    inf_model = Model([user_input, item_input_for_infer], prediction_for_infer, name='generalized_matrix_factorization_for_infer')
    print('########## Model for Inference ###############')
    inf_model.summary()

    return model, inf_model