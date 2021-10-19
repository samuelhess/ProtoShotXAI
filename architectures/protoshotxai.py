from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, TimeDistributed
from tqdm import tqdm

from utils.tensor_operations import *
from utils.distance_functions import *

class ProtoShotXAI:
    def __init__(self, model, input_layer=0, feature_layer=-2, class_layer=-1):
                
        if class_layer != None:
            self.class_weights  = model.layers[class_layer].get_weights()[0]
            self.class_bias     = model.layers[class_layer].get_weights()[1]
        else:
            self.class_weights = None
            self.class_bias = None
        
        input_shape = model.input.shape
        # input_shape = model.layers[input_layer].input_shape[0]
        output_vals = model.layers[feature_layer].output
        model = Model(inputs=model.input, outputs=output_vals)
        
        model_5d = TimeDistributed(model)

        support = Input(input_shape)
        support_features = model_5d(support)
        support_features = Lambda(reduce_tensor)(support_features) 

        query = Input(input_shape)
        query_features = model_5d(query)
        query_features = Lambda(reshape_query)(query_features)

        features = Lambda(cosine_dist_features)([support_features, query_features]) #negative distance
        self.model = Model([support, query], features)
    
    def compute_score_from_features(self,features,iclass):
        s_feature_t, q_feature_t, s_feature_norm, q_feature_norm = features
        s_feature_t = s_feature_t.numpy()
        q_feature_t = q_feature_t.numpy()
        s_feature_t = s_feature_t*np.tile(np.expand_dims(self.class_weights[:,iclass],axis=(0,1)),(s_feature_t.shape[0],s_feature_t.shape[1],1)) 
        q_feature_t = q_feature_t*np.tile(np.expand_dims(self.class_weights[:,iclass],axis=(0,1)),(q_feature_t.shape[0],q_feature_t.shape[1],1))
        s_feature_norm = np.sqrt(np.sum(s_feature_t*s_feature_t,axis=-1))
        q_feature_norm = np.sqrt(np.sum(q_feature_t*q_feature_t,axis=-1))
        den = s_feature_norm * q_feature_norm
        score = np.squeeze(np.sum(s_feature_t*q_feature_t,axis=-1)/den)

        return score

    def image_feature_attribution(self,support_data,query, class_indx, ref_pixel, pad=4 , progress_bar=True):
        rows = np.shape(query)[1]
        cols = np.shape(query)[2]
        chnls = np.shape(query)[3]
        
        query_expand = np.expand_dims(np.copy(query),axis=0) # Batch size of 1
        support_data_expand = np.expand_dims(np.copy(support_data),axis=0) # Only 1 support set

        features = self.model([support_data_expand,query_expand])
        ref_score = self.compute_score_from_features(features,class_indx)

        score_matrix = np.zeros((rows,cols))
        peturbed_images = np.zeros((cols,rows,cols,chnls))
        for ii in tqdm(range(rows),disable=(not progress_bar)):
            for jj in range(cols):
                peturbed_images[jj,:,:,:] = np.copy(query)
                min_ii = np.max([ii-pad,0])
                max_ii = np.min([ii+pad,rows])
                min_jj = np.max([jj-pad,0])
                max_jj = np.min([jj+pad,cols])
                for ichnl in range(chnls):
                    peturbed_images[jj,min_ii:max_ii,min_jj:max_jj,ichnl] = ref_pixel[ichnl]
            
            peturbed_images_expand = np.expand_dims(np.copy(peturbed_images),axis=0)
            features = self.model([support_data_expand,peturbed_images_expand])
            scores = self.compute_score_from_features(features,class_indx)
            score_matrix[ii,:] = ref_score - scores
        
        return score_matrix
        

