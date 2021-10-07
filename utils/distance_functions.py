import tensorflow as tf
import numpy as np

def proto_dist(x):
    feature, pred = x
    n_q = tf.shape(pred)[0]
    n_s = tf.shape(feature)[0]
    s_feature_t = tf.tile(tf.expand_dims(feature,axis = 1), (1,n_q,1))
    pred_t = tf.tile(tf.expand_dims(pred,axis = 0), (n_s,1,1))
    pred_dist = tf.transpose(tf.reduce_sum((s_feature_t-pred_t) ** 2, axis=2))
    return tf.nn.softmax(-(tf.sqrt(pred_dist)))

def eucl_dist(x):
    s_feature, q_feature = x #s_feature (n_way 1, n_support 600,n_features 2048), q_feature (n_query 1, n_features 2048) 
    n_q = tf.shape(q_feature)[0]
    n_w = tf.shape(s_feature)[0]
    n_s = tf.shape(s_feature)[1]
    s_feature_t = tf.reduce_mean(s_feature,axis=1) # average over support set (n_way 1,n_features 2048) 
    s_feature_t = tf.tile(tf.expand_dims(s_feature_t,axis = 1), (1,n_q,1)) #s_feature_t (n_way 1, n_query 1, n_features 2048)
    
    q_feature_t = tf.tile(tf.expand_dims(q_feature,axis = 0), (n_w,1,1)) #q_feature_t (n_way 1, n_query 1, n_features 2048)
    diff = s_feature_t - q_feature_t
    dist = tf.math.sqrt(tf.reduce_sum(tf.multiply(diff,diff),axis=-1))

    return -tf.transpose(tf.squeeze(dist))

def cosine_dist_features(x):
    s_feature, q_feature = x #s_feature (e.g., n_way 1, n_support 600,n_features 2048), q_feature (e.g, n_query 1, n_features 2048) 
    n_q = tf.shape(q_feature)[0]
    n_w = tf.shape(s_feature)[0]
    n_s = tf.shape(s_feature)[1]
    s_feature_t = tf.reduce_mean(s_feature,axis=1) # average over support set (e.g., n_way 1,n_features 2048) 
    s_feature_t = tf.tile(tf.expand_dims(s_feature_t,axis = 1), (1,n_q,1)) #s_feature_t (e.g., n_way 1, n_query 1, n_features 2048)
    s_feature_norm = tf.norm(s_feature_t,axis=-1) #s_feature_t (e.g., n_way 1, n_query 1, n_features 1)
    
    q_feature_t = tf.tile(tf.expand_dims(q_feature,axis = 0), (n_w,1,1)) #q_feature_t (e.g., n_way 1, n_query 1, n_features 2048)
    q_feature_norm = tf.norm(q_feature_t,axis=-1) #s_feature_t (e.g., n_way 1, n_query 1, n_features 1)

    return s_feature_t, q_feature_t, s_feature_norm, q_feature_norm

def cosine_dist(x):
    s_feature_t, q_feature_t, s_feature_norm, q_feature_norm = cosine_dist_features(x)
    
    den = tf.multiply(s_feature_norm,q_feature_norm)
    num = tf.reduce_sum(tf.multiply(s_feature_t,q_feature_t),axis=-1)

    return tf.transpose(tf.squeeze(tf.divide(num,den)))

def cosine_loss(x):
    return tf.nn.softmax(cosine_dist(x))

def post_process_score(x,base_model_weights,iclass):
    s_feature_t, q_feature_t, s_feature_norm, q_feature_norm = x
    s_feature_t = s_feature_t.numpy()
    q_feature_t = q_feature_t.numpy()
    # base_model_weights = np.ones_like(base_model_weights)
    s_feature_t = s_feature_t*np.tile(np.expand_dims(base_model_weights[:,iclass],axis=(0,1)),(s_feature_t.shape[0],s_feature_t.shape[1],1)) 
    q_feature_t = q_feature_t*np.tile(np.expand_dims(base_model_weights[:,iclass],axis=(0,1)),(q_feature_t.shape[0],q_feature_t.shape[1],1))
    s_feature_norm = np.sqrt(np.sum(s_feature_t*s_feature_t,axis=-1))
    q_feature_norm = np.sqrt(np.sum(q_feature_t*q_feature_t,axis=-1))
    den = s_feature_norm * q_feature_norm
    score = np.squeeze(np.sum(s_feature_t*q_feature_t,axis=-1)/den)

    return score

def post_process_return_features(x,base_model_weights,iclass):
    s_feature_t, q_feature_t, s_feature_norm, q_feature_norm = x
    s_feature_t = s_feature_t.numpy()
    q_feature_t = q_feature_t.numpy()
    # base_model_weights = np.ones_like(base_model_weights)
    s_feature_t = s_feature_t*np.tile(np.expand_dims(base_model_weights[:,iclass],axis=(0,1)),(s_feature_t.shape[0],s_feature_t.shape[1],1)) 
    q_feature_t = q_feature_t*np.tile(np.expand_dims(base_model_weights[:,iclass],axis=(0,1)),(q_feature_t.shape[0],q_feature_t.shape[1],1))
    s_feature_norm = np.sqrt(np.sum(s_feature_t*s_feature_t,axis=-1))
    q_feature_norm = np.sqrt(np.sum(q_feature_t*q_feature_t,axis=-1))
    den = s_feature_norm * q_feature_norm
    score = np.squeeze(np.sum(s_feature_t*q_feature_t,axis=-1)/den)

    return s_feature_t, q_feature_t, den


    
    
