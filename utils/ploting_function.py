import matplotlib.pyplot as plt
import utils.shap_color_scheme.colors as colors
from skimage.color import rgb2gray
import numpy as np
from skimage.transform import resize as imresize

def xai_plot(feature_attributions,background,input_percentile=99.9, input_alpha=0.4):
    
    shape_fa  = np.shape(feature_attributions)
    shape_b = np.shape(background)

    if len(shape_fa) > 2:
        feature_attributions = np.squeeze(np.mean(feature_attributions,axis=-1))
    
    if (shape_fa[0] != shape_b[0]) or (shape_fa[1] != shape_b[1]):
        feature_attributions = imresize(feature_attributions, (shape_b[0],shape_b[1]))

    abs_vals = np.abs(feature_attributions).flatten()
    max_val = np.nanpercentile(abs_vals, input_percentile)
    fig = plt.figure(frameon=False)
    im1 = plt.imshow(rgb2gray(background), interpolation = 'nearest', cmap='gray', alpha=input_alpha)
    im2 = plt.imshow(feature_attributions, cmap=colors.red_transparent_blue, vmin=-max_val, vmax=max_val)
    plt.axis('off')
    return plt