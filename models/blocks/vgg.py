import sys
import os
__all__ = [
    "Vgg16LayerBuf_V2",
    "Vgg16LayerBuf_V4",
]
import itertools
from typeguard import typechecked
from typing import List,Union,Tuple
import functools
import tensorflow as tf
import numpy as np 
base = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base,'../'))
sys.path.append(os.path.join(base,'../../'))
from models.craft.convolutions.conv2d import Conv2DVgg
from models.craft.convolutions.conv3d import Vgg2Conv3D
from models.craft.denses import DenseVgg
from training.losses._image_losses import LossAcrossListWrapper
from training.losses._image_losses import MeanFeatureReconstructionError
from training.losses._image_losses import MeanStyleReconstructionError
from training.losses._image_losses import MeanVolumeGradientError
from training.losses._image_losses import MeanAbsoluteError
from training.losses._image_losses import MeanSquaredError
import time

class FeatureMapsGetter(tf.keras.Model):
    @typechecked
    def __init__(self,
                 name:Union[None,str]=None,
                 model_name:str="vgg16",
                 data_format:str="channels_last",
                 use_pooling:bool=False,
                 feature_maps_indicators:Tuple=((0)),
                 **kwargs):
       
        if name is not None:
            name = name+"_"+model_name+"_feature_maps_getter"
        else:
            name = model_name+"_feature_maps_getter"
        super().__init__(name=name,**kwargs)
        _model_name = model_name.lower()
        if _model_name == "vgg16":
            self._model = tf.keras.applications.vgg16.VGG16(
                            include_top=False, weights='imagenet', input_tensor=None,
                            input_shape=None, pooling=None, classes=1000,
                            classifier_activation='softmax'
                            )
            self._preprocess_input = functools.partial(tf.keras.applications.vgg16.preprocess_input,data_format="channels_last")
            self._valid_layer_indexes = list(range(len(self._model.layers)))
            self._inputs_perm_indicator = {}
            self._inputs_perm_indicator["support_data_format"] = "channels_last"
            self._inputs_perm_indicator["batch_index"] = 0
            self._inputs_perm_indicator["channel_index"] = -1
            self._inputs_perm_indicator["meaningful_indexes"] = [-3,-2,-1]
            self._inputs_perm_indicator["reperm_target_indexes"] = [-3,-2]
            self._inputs_perm_indicator["reperm_fixed_indexes"] = [0,-1]

            input_layer_index = 0
            pooling_layer_indexes = [3,6,10,14,18]
            self._valid_layer_indexes.remove(input_layer_index)
            if not use_pooling:
                for index in pooling_layer_indexes:
                    self._valid_layer_indexes.remove(index)
        
        elif _model_name == "vgg19":
            self._model = tf.keras.applications.vgg19.VGG19(
                            include_top=False, weights='imagenet', input_tensor=None,
                            input_shape=None, pooling=None, classes=1000,
                            classifier_activation='softmax'
                            )
            self._preprocess_input = functools.partial(tf.keras.applications.vgg19.preprocess_input,data_format="channels_last")
            self._valid_layer_indexes = list(range(len(self._model.layers)))
            self._inputs_perm_indicator = {}
            self._inputs_perm_indicator["support_data_format"] = "channels_last"
            self._inputs_perm_indicator["batch_index"] = 0
            self._inputs_perm_indicator["channel_index"] = -1
            self._inputs_perm_indicator["meaningful_indexes"] = [-3,-2,-1]

            input_layer_index = 0
            pooling_layer_indexes = [3,6,10,11,16,21]
            self._valid_layer_indexes.remove(input_layer_index)
            if not use_pooling:
                for index in pooling_layer_indexes:
                    self._valid_layer_indexes.remove(index)
        self._model.trainable = False
        self._data_format = data_format.lower() #  inputs' data_format, received by call()
    
        self._feature_maps_indicators = []
        for indicator in feature_maps_indicators:
            self._feature_maps_indicators.append(tuple(sorted(indicator)))
        self._feature_maps_indicators = tuple(self._feature_maps_indicators)

        _max_layer_indexes = [max(x) if len(x)>0 else 0 for x in self._feature_maps_indicators]
        self._fused_index = max(_max_layer_indexes)
   
    def _dist_feature_maps(self,tensor,layer_out_index,feature_maps_vectors):
        """
        put a tensor to feature_maps_vectors
        layer_out_index indicates the tensor is output by which layer, corresponding layer index
        feature_maps_vectors is a 2D list to contain feature_maps
        since
            self._feature_maps_indicators is a 2D tuple of integer
            If layer_out_index in some row of self._feature_maps_indicators,
                it means the tensor is one of target feature map, should be recorded in feature_maps_vectors correspondingly.
        """
        for row_index in range(len(feature_maps_vectors)):
            if layer_out_index in self._feature_maps_indicators[row_index]: # 
                tmp_row = feature_maps_vectors[row_index][:]
                tmp_row.append(tensor)
                feature_maps_vectors[row_index] = tmp_row

        # Since feature_maps_vectors is just a reference, not copy, there is no need to return. But for more easily understanding, still return.
        # return feature_maps_vectors 
    def _normalize_input(self,inputs,**kwargs):
        """
        see https://github.com/keras-team/keras/blob/v2.8.0/keras/applications/vgg16.py#L230-L233
        The images are converted from RGB to BGR, then each color channel is zero-centered with respect to the ImageNet dataset, without scaling.
        """
        inputs = self._normalize_data_format(inputs)
        shape = inputs.shape.as_list()
        shape[-1]=3
        inputs = tf.broadcast_to(inputs,shape)
        assert inputs.shape[-1]==3
        return self._preprocess_input(inputs)
    def _normalize_data_format(self,inputs):
        """
        vgg layers only support channels_last data_format
        even though preprocess_input func support both channels_last and channels_first
        So, after receiving inputs, we compulsorily change inputs's data_format to channels_last if it is channels_first.
        """
        if self._data_format == "channels_first":
            perm = list(range(len(inputs.shape)))
            perm = [perm[0]]+perm[2::]+[perm[1]]
            inputs = tf.transpose(inputs,perm)
        return inputs
    def reduce_high_dimension_inputs(self,inputs):
        """
        Transer a tensor's dimension if its shape different from the vgg needs.
        For a specific vgg layer, input should be [B,H,W,3] in shape.
        `B` dimension  is batch dimension and will be preserved when calculation.
        If a tensor has more than this shape, like [B,D1,D2,D3,...,H,W,3],
        the `B,D1,D2,D3,...` dimension will be considered as general (broad sense) 
        batch dimension and preserved when calculation.

        If we use vgg for `Perceptual Loss`, we take the following steps:
            1. Considier 2 inputs `y_true and y_pred`, both in shape  [B,D1,D2,D3,...,H,W,C] or  [B,C,D1,D2,D3,...,H,W]
                if their data_format is "channels_first", we transpose it to "channels_last"
                then
                y_true's shape is [B,D1,D2,D3,...,H,W,C]
                y_pred's shape is [B,D1,D2,D3,...,H,W,C]
                if C!=3 and C!=1 (C cannot broadcast to 3), raise error since it not suitable to vgg layers
                broadcast `C` dimension
                then
                y_true's shape is [B,D1,D2,D3,...,H,W,3]
                y_pred's shape is [B,D1,D2,D3,...,H,W,3]
            2. Since vgg actually works on last 3 dimensions [H,W,3],
                a smallest input shape is [1,H,W,3] (maybe [1,H,W,1] and broadcast to [1,H,W,3])
                then we will got a list of `n` feature maps:
                    [f1,f2,...,fn], 
                    each `fx` is [1,new_H,new_W,new_C] shape, 
                    where new_H,new_W,new_C are determined by concrete vgg layers.
                So, more generally,
                y_true's feature maps is [f1_true,f2_true,...,fn_true]
                    each `fx` is [B,D1,D2,D3,...,new_H,new_W,new_C] shape,
                y_pred's feature maps is [f1_pred,f2_pred,...,fn_pred]
                    each `fx` is [B,D1,D2,D3,...,new_H,new_W,new_C] shape,
                then, some following loss functions work on the list of feature maps and get a mean loss.
                Hence, if inputs are `High Dimension` tensors, only last 3 dimensions works through vgg layers.
                The finally loss is an average over other dimensions. 
                
                If inputs in 4-dimension shape, like [B,H,W,C], the finally loss will meet our expectations,
                but if inputs in 5 or more dimensions, the output loss will be contrary to our expectations,
                i.e., only last 3 dimensions are considered as 2-D images, unfair to other dimensions (except real batch dimension).

                So, if inputs in high dimension, we take the following steps:
                    1. Considier 2 inputs `y_true and y_pred`, have N dimensions, both in shape [B,D1,D2,...,H,W,3],
                        real batch size (not broad sense batch size) is `B`, channels dimension has broadcast to 3 and transposed to last.
                        make a list of tensor from y_true
                        [y1_true,y2_true,...,y{C_{N-2}^{2}}_true], 
                            "{C_{N-2}^{2}}" means choosing 2 dimensions as "pixel wise" dimension from `D1,D2,...,H,W`
                        
                        make a list of tensor from y_pred
                        [y1_pred,y2_pred,...,y{C_{N-2}^{2}}_pred], 
                            "{C_{N-2}^{2}}" means choosing 2 dimensions as "pixel wise" dimension from `D1,D2,...,H,W`
                        then, we want to compute the loss across above 2 lists ans got a loss list
                        [loss(y1_true,y1_pred),loss(y2_true,y2_pred),...,loss(y{C_{N-2}^{2}}_true,y{C_{N-2}^{2}}_pred)]

                        the average of this loss list is our real expected loss. It takes use of each dimension (except real batch dimension),
                        and makes each dimension (except real batch dimension and channels dimension) fair.
                        
                    2. With the help of broad sense batch dimension, 
                        if y1_true,y2_true,...,y{C_{N-2}^{2}}_true have the same shape
                        and y1_pred,y2_pred,...,y{C_{N-2}^{2}}_pred have the same shape
                        we can
                            stack [y1_true,y2_true,...,y{C_{N-2}^{2}}_true] as new tensor new_y_true
                            stack [y1_pred,y2_pred,...,y{C_{N-2}^{2}}_pred] as new tensor new_y_pred
                            so we have:
                            loss(new_y_true,new_y_pred) == mean([loss(y1_true,y1_pred),loss(y2_true,y2_pred),...,loss(y{C_{N-2}^{2}}_true,y{C_{N-2}^{2}}_pred)])
                            this is a speedup method.
                        but in most time, y1_true,y2_true,...,y{C_{N-2}^{2}}_true dot not have the same shape
                            
        """
        # suppose inputs' data_format has ben transposed to "channels_last"
        if len(inputs.shape)>=5:
            inputs_list = self._tensor_repermutation(
                            inputs,perm_target_indexes=self._inputs_perm_indicator["reperm_target_indexes"],
                            perm_fixed_indexes=self._inputs_perm_indicator["reperm_fixed_indexes"])
            
            _reduce_general_batch_size = functools.partial(self._reduce_general_batch_size,
                                perm_meaningful_indexes=self._inputs_perm_indicator["meaningful_indexes"])
            inputs_list = list(map(_reduce_general_batch_size,inputs_list))

            try:
                new_inputs = tf.stack(inputs_list,axis=1) # if elements in inputs_list have the same shape
            except (ValueError,tf.errors.InvalidArgumentError):
                new_inputs = inputs_list
            return new_inputs
        else:
            return inputs
    def _reduce_general_batch_size(self,tensor,perm_meaningful_indexes=[]):
        """
        Reshape a tensor, make its dimensions meaningful and work in posterior steps.
        Consider a tensor in shape [B,D1,D2,...D{N-2},C], its meaningful dimensions 
        are `D{N-3}`, `D{N-2}` and `C`.

        If in in posterior steps, such as "get gram matrix", we should consider "[B,D1,D2,..,D{N-4}]"
        as broad sense batch dimension. But if we change nothing, a traditional  "get gram matrix"
        procedure will consider `B` as batch dimension, and meaningful dimensions will be `[D1,D2,...D{N-2}]`,
        leading to the results that far from our excepted.
        So, reducing a tensor's broad sense batch dimensions (general batch dimensions) 
        ,according its meaningful dimensions, is needed.
        """
        N = len(tensor.shape)
        perm_meaningful_indexes = [i+N if i<0 else i for i in perm_meaningful_indexes] # normalize index
        perm_meaningful_indexes = sorted(perm_meaningful_indexes)
        perm = [i for i in range(N) if i not in perm_meaningful_indexes]
        perm = perm+perm_meaningful_indexes
        remaining_shape = [tensor.shape[i] for i in perm_meaningful_indexes]
        return tf.reshape(tf.transpose(tensor,perm),[-1]+remaining_shape)

    def _tensor_repermutation(self,tensor,perm_target_indexes=[],perm_fixed_indexes=[]):
        """
        Re-permutation a tensor by its perm's permutation and combination.
        Consider a N dimensions tensor in shape [B,D1,D2,...D{N-2},C] 
        its perm (perm_list|perm_index_list) is [0,1,2,...,N-1], i.e., perm_list is [0,1,2,...,N-1]
        The following steps explain how to make perm's permutation and combination to achieve our goal:
            1. from perm_fixed_indexes to find un-fixed perm indexes
                if we fix `B`, `C` dimension, perm_fixed_indexes==[0,-1] 
                the un-fixed dimensions are `D1,D2,...D{N-2}`, their indexes are [1,2,...,{N-2}] 
            2. choose n=len(perm_target_indexes) elements from  un-fixed perm indexes, without change their relevant order.
                we get selected 'n' elements with original order
                we get un-selected 'N-len(perm_fixed_indexes)-n' elements with original order
                if un-fixed indexes are [1,2,...,{N-2}] , perm_target_indexes=[-3,-2]
                    selected 'n' elements will be:
                        [1,2] 
                        [1,3]
                        ...
                        [N-3,N-2]
                    un-selected 'N-len(perm_fixed_indexes)-n' elements will be:
                        [3,4,...,N-2]
                        [2,4,...,N-2]
                        ...
                        [1,2,...,N-4]
                        correspondingly
            3.  then, forcus on perm list's un-fixed indexes 
                clear perm list's un-fixed indexes, 
                put selected 'n' elements along perm_target_indexes correspondingly
                put un-selected 'N-len(perm_fixed_indexes)-n' elements to the remaining position
                if un-fixed indexes are [1,2,...,{N-2}] , perm_target_indexes=[-3,-2]
                    cleared un-fixed indexes are [None,None,...,None]
                    put selected and un-selected elements, then we will get the final perm:
                        [0,3,4,...,N-2,1,2,N-1]
                        [0,2,4,...,N-2,1,3,N-1]
                        ...
                        [0,1,2,...,N-4,N-3,N-2,N-1]
                    So perm's permutation and combination has C_{N-len(perm_fixed_indexes)}^{len(perm_target_indexes)} results.
                    Each result leads to a excepted re-permutation of tensor。
            Final output is a list of these excepted re-permutation tensors.
        Args:
            tensor: the original tensor
            perm_target_indexes: list of integer
            perm_fixed_indexes: list of integer
        Return 
            Re-permutation result

        """
        # [B,D1,D2,...,H,W,3] N dimensions
        input_shape = tensor.shape # y_true or y_pred
        N = len(input_shape)
        perm_target_indexes = [i+N if i<0 else i for i in perm_target_indexes ] # normalize index
        perm_fixed_indexes = [i+N if i<0 else i for i in perm_fixed_indexes] # normalize index

        _confused_indexes = [i for i in perm_target_indexes if i in perm_fixed_indexes]
        assert len(_confused_indexes)==0

        perm_all_indexes = list(range(N))
        _perm_fixed_indexes = sorted(perm_fixed_indexes,reverse=True)
        for index in _perm_fixed_indexes:
            perm_all_indexes.pop(index) # pop() will work since index and value are equal
        unfixed_indexes = perm_all_indexes
        perms_buf = []
        for selected_indexes in itertools.combinations(unfixed_indexes,len(perm_target_indexes)):
            perm = [None,]*N
            for index in perm_fixed_indexes:
                perm[index] = index 
            for source_index,target_index in zip(selected_indexes,perm_target_indexes):
                perm[target_index] = source_index 
            unselected_indexes = unfixed_indexes[:]
            for index in selected_indexes:
                unselected_indexes.remove(index) # pop() will not work since index and value are not equal
            remaining_indexes = [i for i,x in enumerate(perm) if x is None ]
            for source_index,target_index in zip(unselected_indexes,remaining_indexes):
                perm[target_index] = source_index 
            perms_buf.append(perm)
        return [tf.transpose(tensor,perm) for perm in perms_buf]
    def call(self,inputs,**kwargs):
        x = self._normalize_input(inputs)
        output_buf = [[],]*len(self._feature_maps_indicators)
        for index in self._valid_layer_indexes:
            _layer  = self._model.get_layer(index=index)
            y = _layer(x,**kwargs)
            self._dist_feature_maps(tensor=y,layer_out_index=index,feature_maps_vectors=output_buf)
            x = y
            if index >= self._fused_index:
                break
        return output_buf

class PerceptualLossExtractor(tf.keras.Model):
    @typechecked
    def __init__(self,
                 name:Union[None,str]=None,
                 model_name:str="vgg16",
                 data_format:str="channels_last",
                 valid_layer_index:List[int]=[1,2,4,5,7,8,9,11,12,13,15,16,17],
                 use_feature_reco_loss:bool=True,
                 use_style_reco_loss:bool=False,
                 feature_reco_index:List[int]=[5],
                 feature_reco_sample_weight:List[int]=[1],
                 style_reco_index:List[int]=[2,5,9,13],
                 style_reco_sample_weight:List[int]=[1,1,1,1],
                 **kwargs):
        if name is not None:
            name = name+"_"+model_name+"_perceptual_loss_extractor"
        else:
            name = model_name+"_perceptual_loss_extractor"
        # kwargs["dtype"] = None # vgg16 should work in float32 as default
        super(PerceptualLossExtractor,self).__init__(name=name,**kwargs)
       
        self.data_format = data_format.lower() #  inputs' data_format, received by call()
        if self.data_format not in ["channels_first","channels_last"]:
            raise ValueError("data_format of PerceptualLoss should be 'channels_last' or 'channels_first', not {}.".format(data_format))
        self.valid_layer_index = valid_layer_index
        self.use_feature_reco_loss = use_feature_reco_loss
        self.use_style_reco_loss  = use_style_reco_loss
        feature_reco_loss = MeanFeatureReconstructionError(mode="L2")
        self.feature_reco_loss = LossAcrossListWrapper(feature_reco_loss)
        self.feature_reco_index = feature_reco_index # relu3_3
        self.feature_reco_sample_weight = feature_reco_sample_weight
        assert (len(self.feature_reco_index)==len(self.feature_reco_sample_weight))
        style_reco_loss = MeanStyleReconstructionError(data_format="channels_last")
        self.style_reco_loss = LossAcrossListWrapper(style_reco_loss)
        self.style_reco_index = style_reco_index # relu1_2 relu2_2 relu3_3 relu4_3
        self.style_reco_sample_weight = style_reco_sample_weight
        assert (len(self.style_reco_index)==len(self.style_reco_sample_weight))
       
        
        feature_maps_indicators=(tuple(feature_reco_index) if use_feature_reco_loss else tuple([]),tuple(style_reco_index)  if use_style_reco_loss else tuple([]))
        self.feature_maps_getter = FeatureMapsGetter(name=name,
                                    model_name=model_name,
                                    data_format=data_format,
                                    use_pooling=False,
                                    feature_maps_indicators=feature_maps_indicators,
                                    **kwargs)
        self._preprocess_input = self.feature_maps_getter._preprocess_input
    def _normalize_data_format(self,inputs):
        """
        vgg layers only support channels_last data_format
        even though preprocess_input func support both channels_last and channels_first
        So, after receiving inputs, we compulsorily change inputs's data_format to channels_last if it is channels_first.
        """
        if self.data_format == "channels_first":
            perm = list(range(len(inputs.shape)))
            perm = [perm[0]]+perm[2::]+[perm[1]]
            inputs = tf.transpose(inputs,perm)
        return inputs
    def get_feature_maps(self,inputs,feature_reco_sample_weight,style_reco_sample_weight,**kwargs):
        if isinstance(inputs,list):
            _featuere_maps_for_feat = []
            _featuere_maps_for_style = []
            _feature_reco_sample_weight = []
            _style_reco_sample_weight = []
            for single_input in inputs:
                featuere_maps_for_feat,featuere_maps_for_style = self.feature_maps_getter(single_input,**kwargs)
                _featuere_maps_for_feat += featuere_maps_for_feat
                _featuere_maps_for_style += featuere_maps_for_style
                _feature_reco_sample_weight += feature_reco_sample_weight
                _style_reco_sample_weight += style_reco_sample_weight
            featuere_maps_for_feat = _featuere_maps_for_feat
            featuere_maps_for_style = _featuere_maps_for_style
            feature_reco_sample_weight = _feature_reco_sample_weight
            style_reco_sample_weight = _style_reco_sample_weight
        else:
            featuere_maps_for_feat,featuere_maps_for_style = self.feature_maps_getter(inputs,**kwargs)
        return (featuere_maps_for_feat,featuere_maps_for_style,feature_reco_sample_weight,style_reco_sample_weight)
    def build(self,input_shape):
        super().build(input_shape)
        # input_shape = tf.TensorShape(input_shape)
        # assert input_shape[0]==2
        # self.input_spec = tf.keras.layers.InputSpec(shape=input_shape)
    def summary(self):
        self.model.summary()
    def call(self,inputs,**kwargs):
        # inputs_true,inputs_pred = self.normalize_input(inputs[0]),self.normalize_input(inputs[1])
        inputs_true,inputs_pred = self.feature_maps_getter.reduce_high_dimension_inputs(inputs[0]),self.feature_maps_getter.reduce_high_dimension_inputs(inputs[1])
        kwargs["training"] = False # force un-training
        # print(inputs_true.shape)
        loss = 0.0
        if (self.use_feature_reco_loss)or(self.use_style_reco_loss):
            true_featuere_maps_for_feat,true_featuere_maps_for_style,\
                true_feature_reco_sample_weight,true_style_reco_sample_weight =\
                    self.get_feature_maps(inputs_true,self.feature_reco_sample_weight,self.style_reco_sample_weight,**kwargs)
            pred_featuere_maps_for_feat,pred_featuere_maps_for_style,\
                pred_feature_reco_sample_weight,pred_style_reco_sample_weight =\
                    self.get_feature_maps(inputs_pred,self.feature_reco_sample_weight,self.style_reco_sample_weight,**kwargs)
            assert (feature_reco_sample_weight:=true_feature_reco_sample_weight)==pred_feature_reco_sample_weight
            assert (style_reco_sample_weight:=true_style_reco_sample_weight)==pred_style_reco_sample_weight
            if self.use_feature_reco_loss:
                feature_reco_loss = self.feature_reco_loss(
                                        true_featuere_maps_for_feat,
                                        pred_featuere_maps_for_feat,
                                        feature_reco_sample_weight)
                loss += feature_reco_loss
            if self.use_style_reco_loss:
                style_reco_loss = self.style_reco_loss(
                                        true_featuere_maps_for_style,
                                        pred_featuere_maps_for_style,
                                        style_reco_sample_weight)
                loss += style_reco_loss              
        return loss
#------------------------------------------------------------------------------------------#
class Vgg16LayerBuf_V4(tf.keras.Model):
    # 输出VGG16的前指定若干层
    def __init__(self,path="D:\\Datasets\\VGG\\vgg16.npy",
                 name=None,
                 dtype=None):
        self.path = path 
        self.data_dict = np.load(self.path,encoding='latin1',allow_pickle=True).item()
        super(Vgg16LayerBuf_V4,self).__init__(name=name,dtype=dtype)
        self.conv1_1 = Vgg2Conv3D(filters=64,kernel_size=[1,3,3],strides=[1,1,1],padding="SAME",use_bias=True,activation=None,dtype=dtype)
        self.conv1_2 = Vgg2Conv3D(filters=64,kernel_size=[1,3,3],strides=[1,1,1],padding="SAME",use_bias=True,activation=None,dtype=dtype)
        # self.l1_max_pool = tf.keras.layers.MaxPool2D(pool_size=[2,2], strides=[2,2], padding='valid')
        self.conv2_1 = Vgg2Conv3D(filters=128,kernel_size=[1,3,3],strides=[1,1,1],padding="SAME",use_bias=True,activation=None,dtype=dtype)
        self.conv2_2 = Vgg2Conv3D(filters=128,kernel_size=[1,3,3],strides=[1,1,1],padding="SAME",use_bias=True,activation=None,dtype=dtype)

        self.conv3_1 = Vgg2Conv3D(filters=256,kernel_size=[1,3,3],strides=[1,1,1],padding="SAME",use_bias=True,activation=None,dtype=dtype)
        self.conv3_2 = Vgg2Conv3D(filters=256,kernel_size=[1,3,3],strides=[1,1,1],padding="SAME",use_bias=True,activation=None,dtype=dtype)
        self.conv3_3 = Vgg2Conv3D(filters=256,kernel_size=[1,3,3],strides=[1,1,1],padding="SAME",use_bias=True,activation=None,dtype=dtype)
    def build(self,input_shape):
        input_shape = [None,1,224,224,3]
        flow_shape=self.conv1_1.build(input_shape=input_shape)
        flow_shape=self.conv1_2.build(input_shape=flow_shape) 
        flow_shape[-3]//=2
        flow_shape[-2]//=2
        flow_shape=self.conv2_1.build(input_shape=flow_shape) 
        flow_shape=self.conv2_2.build(input_shape=flow_shape) 
        flow_shape[-3]//=2
        flow_shape[-2]//=2
        flow_shape=self.conv3_1.build(input_shape=flow_shape) 
        flow_shape=self.conv3_2.build(input_shape=flow_shape) 
        flow_shape=self.conv3_3.build(input_shape=flow_shape)
        output_shape = flow_shape[:]
     
        self.conv1_1.w.assign(tf.reshape(self.data_dict["conv1_1"][0],[1,3,3,3,64]))
        self.conv1_1.b.assign(self.data_dict["conv1_1"][1])
        self.conv1_2.w.assign(tf.reshape(self.data_dict["conv1_2"][0],[1,3,3,64,64]))
        self.conv1_2.b.assign(self.data_dict["conv1_2"][1])
        self.conv2_1.w.assign(tf.reshape(self.data_dict["conv2_1"][0],[1,3,3,64,128]))
        self.conv2_1.b.assign(self.data_dict["conv2_1"][1])
        self.conv2_2.w.assign(tf.reshape(self.data_dict["conv2_2"][0],[1,3,3,128,128]))
        self.conv2_2.b.assign(self.data_dict["conv2_2"][1])
        self.conv3_1.w.assign(tf.reshape(self.data_dict["conv3_1"][0],[1,3,3,128,256]))
        self.conv3_1.b.assign(self.data_dict["conv3_1"][1])
        self.conv3_2.w.assign(tf.reshape(self.data_dict["conv3_2"][0],[1,3,3,256,256]))
        self.conv3_2.b.assign(self.data_dict["conv3_2"][1])
        self.conv3_3.w.assign(tf.reshape(self.data_dict["conv3_3"][0],[1,3,3,256,256]))
        self.conv3_3.b.assign(self.data_dict["conv3_3"][1])

        return output_shape
    def call(self,x,training=True,scale=4):
        layer_buf = []
        x = tf.broadcast_to(x,x.shape[0:-1]+[3])
        x = x/3
        layer_buf.append(x)
        if scale < 1:
            return layer_buf
        x=self.conv1_1(x,training=training)
        layer_buf.append(x)
        if scale < 2:
            return layer_buf
        x=self.conv1_2(x,training=training)
        layer_buf.append(x)
        if scale < 3:
            return layer_buf 
        x=self.conv2_1(x,training=training)
        layer_buf.append(x)
        if scale < 4:
            return layer_buf
        x=self.conv2_2(x,training=training)
        layer_buf.append(x)
        if scale < 5:
            return layer_buf
        x=self.conv3_1(x,training=training)
        layer_buf.append(x)
        if scale < 6:
            return layer_buf
        x=self.conv3_2(x,training=training)
        layer_buf.append(x)
        if scale < 7:
            return layer_buf
        x=self.conv3_3(x,training=training)
        layer_buf.append(x)
        if scale < 8:
            return layer_buf
        else:
            raise ValueError("Layers Must Under 7")

if __name__ == "__main__":
    physical_devices = tf.config.experimental.list_physical_devices(device_type='GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    tf.keras.utils.set_random_seed(1)
    tf.config.experimental.enable_op_determinism()
    policy = tf.keras.mixed_precision.Policy('float32')
    import time
    x = tf.random.uniform(shape=[1,128,128,16]) 
    x_ = tf.reshape(x,[1,128,128,16,1])
    x_ = tf.broadcast_to(x_,[1,128,128,16,3])
    print("********************V4 start******************")
    V = Vgg16LayerBuf_V4(dtype=policy)
    V.build(input_shape=None)
    x_ = tf.transpose(x_,perm=[0,3,1,2,4])
    m = [tf.keras.metrics.Mean() for _ in range(5)]
    start = time.time()
    for _ in range(100):
        for index in range(5):
            m[index].reset_states()
        for index,y in enumerate(V(x_)):
            m[index](y)
    print(time.time()-start)
    for index in range(5):
        print(m[index].result().numpy())
    start = time.time()
    for _ in range(100):
        for index in range(5):
            m[index].reset_states()
        for index,y in enumerate(V(x_)):
            m[index](y)
    print("time",time.time()-start)
    for index in range(5):
        print(m[index].result().numpy())
    print("********************V4 end******************")

    print("********************V5 start******************")
    feature_maps_indicators = ((1,2,4,5),())
    feature_maps_getter = FeatureMapsGetter(use_pooling=False,feature_maps_indicators=feature_maps_indicators,dtype=policy)
    # V.build(input_shape=None)
    # x_ = tf.transpose(x_,perm=[0,3,1,2,4])
    m = [tf.keras.metrics.Mean() for _ in range(5)]
    start = time.time()
    for _ in range(100):
        for index in range(5):
            m[index].reset_states()
        for index, y in enumerate(feature_maps_getter(x_)[0][0:4]):
            # print(len(y))
            m[index](y)
    print(time.time()-start)
    for index in range(5):
        print(m[index].result().numpy())
    start = time.time()
    for _ in range(100):
        for index in range(5):
            m[index].reset_states()
        for index,y in enumerate(feature_maps_getter(x_)[0][0:4]):
            m[index](y)
    print("time",time.time()-start)
    for index in range(5):
        print(m[index].result().numpy())
    print("********************V5 end******************")



    vf = PerceptualLossExtractor(valid_layer_index=[1,2,4,5,7,8,9,11,12,13,15,16,17],dtype="mixed_float16")
    m =tf.keras.metrics.Mean() 
    start = time.time()
    for _ in range(100):
        m.reset_states()
        y=vf(inputs=[x_,x_])
        m(y)
    print(time.time()-start)
    print(m.result().numpy())
    start = time.time()
    for _ in range(100):
        m.reset_states()
        y=vf(inputs=[x_,x_])
        m(y)
    print("time",time.time()-start)
    print(m.result().numpy())
    print("**************************************")

    # y_true = tf.stack([_y_true,_y_true])
    # y_pred = tf.stack([_y_pred,_y_pred])
    # y_true = tf.transpose(y_true,perm=[1,0,2,3,4])
    # y_pred = tf.transpose(y_pred,perm=[1,0,2,3,4])
    # vf.build(input_shape=[2,4,2,128,128,3])
    # # vf.build(input_shape=[2,2,4,128,128,3])
    # y = vf(inputs=tf.stack([y_true,y_pred]))
    # print(y)


   
