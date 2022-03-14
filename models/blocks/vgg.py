import sys
import os
__all__ = [
    "Vgg16LayerBuf_V2",
    "Vgg16LayerBuf_V4",
]
import itertools
import logging
from typeguard import typechecked
from typing import List,Union,Tuple,Iterable
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
    """
    Get feature maps of an input based on an existing linear structure model.
    `linear structure` means we can easily locate the desired layers' feature map output by given index.
    This class usually used when transfer leraning.
    The two main things this class does are normalize input, making it suitable to basic model and 
    collect feature maps as user indicates.

    Args:
        name: name
        model_name: specify the basic linear structure model, currently, only support vgg16 and vgg19.
        data_format: indicate input's data_format, supporting "channels_last" or "channels_first". 
        use_pooling:
            If False, pooling layers in basic model will be skipped when inference.
            If True, pooling layers in basic model will be used as normal when inference.
                Usually, if occured with huge amount of calculation, use_pooling should be True.
        feature_maps_indicators: a 2-D tuple of integer, declare the locations (which layers' outputs) in basic model to give out feature maps
                    e.g. feature_maps_indicators=((1,2),(1,3),(2,4)),
                    It means we will get 3 different list of feature maps, and they will be combined into one list and outputed simultaneously.
                    The first list of feature maps consists of basic linear structure model's 2nd (index 1) layer's output and 3rd (index 2) layer's output.
                    The second list of feature maps consists of basic linear structure model's 2nd (index 1) layer's output and 4th (index 3) layer's output.
                    The third list of feature maps consists of basic linear structure model's 3rd (index 2) layer's output and 5th (index 4) layer's output.
                    Since this is 2-D tuple of integer, we can get any number of feature map lists and any number of feature maps, as long as  basic linear structure model supports.
        other kwargs: dtype cannot be customized currently, since the basic model cannot be serialized and deserialized easily. TODO support specifying dtype.
    """
    @typechecked
    def __init__(self,
                 name:Union[None,str]=None,
                 model_name:str="vgg16",
                 data_format:str="channels_last",
                 use_pooling:bool=False,
                 feature_maps_indicators:Tuple[Tuple,...]=((0,),),
                 **kwargs):
        if name is not None:
            name = name+"_"+model_name+"_feature_maps_getter"
        else:
            name = model_name+"_feature_maps_getter"
        if "dtype" in kwargs.keys():
            if kwargs["dtype"] is not None:
                logging.warning("""
                Setting FeatureMapsGetter's dtype to a specific dtype but not `None` may fail to meet the user's expectations. Since the actually dtype should follow 
                model's practical dtype. For numerical stability, we mandatorily set dtype to None. 
                """)
        kwargs["dtype"] = None
        super().__init__(name=name,**kwargs)

        _model_name = model_name.lower()
        if _model_name == "vgg16":
            self._model = tf.keras.applications.vgg16.VGG16(
                            include_top=False, weights='imagenet', input_tensor=None,
                            input_shape=None, pooling=None, classes=1000,
                            classifier_activation='softmax'
                            )
            self._preprocess_input = functools.partial(tf.keras.applications.vgg16.preprocess_input,data_format="channels_last")
            self._model_meta_data = {}
            self._model_meta_data["input_layer_index"] = 0
            self._model_meta_data["pooling_layer_indexes"] = [3,6,10,14,18]
            self._model_meta_data["forced_pooling_threshold"] = 6 
            self._model_meta_data["supported_data_format"] = "channels_last"
            self._model_meta_data["supported_data_shape"] = [None,None,None,3] 
        elif _model_name == "vgg19":
            self._model = tf.keras.applications.vgg19.VGG19(
                            include_top=False, weights='imagenet', input_tensor=None,
                            input_shape=None, pooling=None, classes=1000,
                            classifier_activation='softmax'
                            )
            self._preprocess_input = functools.partial(tf.keras.applications.vgg19.preprocess_input,data_format="channels_last")
            self._model_meta_data = {}
            self._model_meta_data["input_layer_index"] = 0
            self._model_meta_data["pooling_layer_indexes"] = [3,6,11,16,21]
            self._model_meta_data["forced_pooling_threshold"] = 6
            self._model_meta_data["supported_data_format"] = "channels_last"
            self._model_meta_data["supported_data_shape"] = [None,None,None,3]
        else:
            raise ValueError("{} hasn't been supported currently.".format(model_name))
        self._model.trainable = False

        self._additional_meta_data = {}
        self._data_format = data_format.lower() #  inputs' data_format, received by call()
        self._normalize_input_data_format = self._normalize_input_data_format_wrapper(
            self._data_format,
            self._model_meta_data["supported_data_format"])
            
        self._additional_meta_data["batch_index"] = 0 # the precondition
        if self._data_format =="channels_last":
            self._additional_meta_data["channel_index"] = -1
            self._additional_meta_data["reperm_target_indexes"] = [-3,-2] # H,W in [B,...,H,W,C] 
        else:
            self._additional_meta_data["channel_index"] = 1
            self._additional_meta_data["reperm_target_indexes"] = [-2,-1] # H,W in [B,C,...,H,W]
        self._additional_meta_data["reperm_fixed_indexes"] = [self._additional_meta_data["batch_index"],self._additional_meta_data["channel_index"]] # B,C
        self._additional_meta_data["reshape_fixed_indexes"] = self._additional_meta_data["reperm_target_indexes"]+[self._additional_meta_data["channel_index"]] # H W C
        
        self._feature_maps_indicators,_layer_indexes_set = self._sort_indicators(feature_maps_indicators)
        self._fused_index = max(_layer_indexes_set)

        use_pooling = self._check_if_forced_use_pooling(
            concerned_index=self._fused_index,
            threshold=self._model_meta_data["forced_pooling_threshold"],
            use_pooling=use_pooling)

        self._additional_meta_data["valid_layer_indexes"] = self._grab_valid_indexes(
            original_indexes=list(range(len(self._model.layers))),
            invalid_indexs=[self._model_meta_data["input_layer_index"],
                            [] if use_pooling else self._model_meta_data["pooling_layer_indexes"]])

        self._check_if_within_valid_indexes(
            target_indexes=_layer_indexes_set ,
            valid_indexes=self._additional_meta_data["valid_layer_indexes"])

        
    def _sort_indicators(self,indicators:Tuple[Tuple,...]):
        """
        sort indicators (a 2D tuple of integer)
        additionally, give out a sorted set of all elements in the indicators
        """
        indicators_buf = []
        element_set = set()
        for indicator in indicators:
            indicators_buf.append(tuple(sorted(indicator)))
            for item in indicator:
                element_set.add(item)
        return tuple(indicators_buf),sorted(element_set)
    @typechecked
    def _check_if_forced_use_pooling(self,concerned_index:int,threshold:int,use_pooling:bool):
        if concerned_index >= threshold:
            # should forced use pooling
            if use_pooling:
                return use_pooling
            else:
                logging.warning(
                """
                To avoid huge amount of calculation, when the index of layer, which give out wanted feature map, reaches or exceed the index threshold {},
                the basic model will use pooling mandatorily, and `use_pooling` flag set by user will be ignored.
                """.format(threshold))
                return True
        else:
            # no need to forced use pooling
            return use_pooling
    @typechecked
    def _grab_valid_indexes(self,original_indexes:List[int],invalid_indexs:Iterable):
        def _flatten_complex_iter_rable_to_list_of_int(inputs):
            buf = set()
            for item in inputs:
                if isinstance(item,int):
                    buf.add(item)
                elif isinstance(item,Iterable):
                    for inner_item in _flatten_complex_iter_rable_to_list_of_int(item):
                        buf.add(inner_item)
                else:
                    raise ValueError("item should be an iterable object or integer, not {}.".format(item))
            return sorted(buf)
        original_indexes = original_indexes[:]
        invalid_indexs = _flatten_complex_iter_rable_to_list_of_int(invalid_indexs)
        for index in invalid_indexs:
            original_indexes.remove(index)
        return original_indexes
    @typechecked
    def _check_if_within_valid_indexes(self,target_indexes:Iterable,valid_indexes:Iterable):
        for index in target_indexes:
            if index not in valid_indexes:
                raise ValueError("index `{}` in  are not in valid indexes.".format(index))
    def _dist_tensor_to_2D_container(self,tensor,index,indicator,container):
        """
        According to the cooperation between index and 2-D indicator,
        put a tensor to each row of a 2D buffer, i.e., a 2-D list, which can be regarded as a container or vector.

        `tensor` and `index` are one-to-one correspondence.
        If index in indicator's `a` row and `b` column, container's `a` row will append the `tensor`.

        i.e.,`tensor` will be repeated in each row of container if the corresponding `index` is in indicator.
        """
        container = container[:] # copy
        for row_index in range(len(container)):
            if index in indicator[row_index]: # 
                tmp_row = container[row_index][:]
                tmp_row.append(tensor)
                container[row_index] = tmp_row
        return container
    def _normalize_input(self,inputs):
        """
        Normalize input, make it suitable for base model. 
        Generally, a base model provide preprocess_input() function to preprocess input.
        But usually, modify the data_format and broadcast some dimension are also needed.
        For example,
            see https://github.com/keras-team/keras/blob/v2.8.0/keras/applications/vgg16.py#L230-L233
            The images are converted from RGB to BGR, then each color channel is zero-centered with respect to the ImageNet dataset, without scaling.
        normalize input in 4 steps:
            1. normalize data format
            2. reshape more than expected dimensions to batch dimension as broad sense batch dimensions.
                if base model is a fully convolutional model (no pooling layer), there is no need to reshape, since
                 convolution operation takes more than expected dimensions as broad sense batch dimensions,
                 as long as the expected dimensions are one to one correspondence.
                 i.e. convolution operation has equivalence between "reshape then convolution" and "convolution then reshape".
                However, if the base model has pooling layer, we must reshape input tensor, because 
                pooling operation has no equivalence between "reshape then pooling" and "pooling then reshape".
                More than expected dimensions will not be considered as broad sense batch dimensions but meaningful dimensions, in pooling operation.

                Additional, if a posterior operation deals with feature maps given by this class, such as "Gram Matrix", reshape more than expected dimensions is also needed,
                beacuse even though the meaninful dimensions are equivalent before the feature maps, they are not equivalent in feature maps since basic model can not 
                treat each meaninful dimension fairly.

                So, for stability, reshape in needed.
            3. broadcast
            4. preprocess input by base model's preprocess_input function
        """
        inputs = self._normalize_input_data_format(inputs)  # then the channels dimension index will be the same to self._meta_data["channel_index"]
        inputs = self._reshape_and_keep_fixed_dimensions(inputs,self._additional_meta_data["reshape_fixed_indexes"]) # get supported_data_shape
        shape = inputs.shape.as_list()
        for i,(in_shape,ex_shape) in enumerate(zip(shape,self._model_meta_data["supported_data_shape"])): # ex_shape expected_shape
            if ex_shape is not None:
                if in_shape != ex_shape:
                    shape[i] = ex_shape
        inputs = tf.broadcast_to(inputs,shape)
        return self._preprocess_input(inputs)
    def _normalize_input_data_format_wrapper(self,data_format,supported_data_format):
        """
        Usually, a basic model only supports channels_last or channels_first data_format.
        For example,
            vgg layers only support channels_last data_format
            even though its preprocess_input func support both channels_last and channels_first
            So, after receiving inputs, we compulsorily change inputs's data_format.
        """
        def _first_to_last(inputs):
            perm = list(range(len(inputs.shape)))
            perm = [perm[0]]+perm[2::]+[perm[1]]
            inputs = tf.transpose(inputs,perm)
            return inputs 
        def _last_to_first(inputs):
            perm = list(range(len(inputs.shape)))
            perm = [perm[0]]+[perm[-1]]+perm[1:-1]
            inputs = tf.transpose(inputs,perm)
            return inputs 
        def _do_nothing(inputs):
            return inputs
        if data_format != supported_data_format:
            if data_format  == "channels_first":
                return _first_to_last
            elif data_format  == "channels_last":
                return _last_to_first
            else:
                raise ValueError("Data_format should be one of channels_first or channels_last, not {}.".format(data_format))
        else:
            return _do_nothing
        
    def _reshape_and_keep_fixed_dimensions(self,tensor,fixed_dimensions_indexes=[]): # default [] is OK since the function works on its copy
        """
        Reshape a tensor, maintain fixed dimensions, merge unfixed dimensions.
            Usually application is considering unfixed dimensions as broad sense batch dimensions and merge them to real batch dimension.
        If fixed dimensions is discontinuous and not last dimensions, we will transpose it first. So there is no deed to care about data_format, as long as fixed_dimensions_indexes indicates correct indexes. 
        """
        N = len(tensor.shape)
        fixed_dimensions_indexes = sorted([i+N if i<0 else i for i in fixed_dimensions_indexes]) # normalize index
        perm = [i for i in range(N) if i not in fixed_dimensions_indexes]
        perm = perm+fixed_dimensions_indexes
        remaining_shape = [tensor.shape[i] for i in fixed_dimensions_indexes]
        return tf.reshape(tf.transpose(tensor,perm),[-1]+remaining_shape)

    def _tensor_repermutation(self,tensor,reperm_target_indexes=[],reperm_fixed_indexes=[]): # default [] is OK since the function works on its copy
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
                        [0,3,4,...,N-2,{1},{2},N-1]
                        [0,2,4,...,N-2,{1},{3},N-1]
                        ...
                        [0,1,2,...,N-4,{N-3},{N-2},N-1]
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
        reperm_target_indexes = sorted([i+N if i<0 else i for i in reperm_target_indexes]) # normalize index
        reperm_fixed_indexes = sorted([i+N if i<0 else i for i in reperm_fixed_indexes]) # normalize index

        _confused_indexes = [i for i in reperm_target_indexes if i in reperm_fixed_indexes]
        assert len(_confused_indexes)==0

        perm = list(range(N))
        unfixed_indexes = perm[:]
        undetermined_indexes = perm[:]
        perm = [None,]*N
        for index in reperm_fixed_indexes:
            perm[index] = index 
            unfixed_indexes.remove(index)
            undetermined_indexes.remove(index)
        perms_buf = []
        for selected_indexes in itertools.combinations(unfixed_indexes,len(reperm_target_indexes)):
            tmp_perm = perm[:]
            unselected_indexes = unfixed_indexes[:]
            remaining_indexes = undetermined_indexes[:]
            for source_index,target_index in zip(selected_indexes,reperm_target_indexes): # selected_index will be repermutated to target_index
                tmp_perm[target_index] = source_index 
                unselected_indexes.remove(source_index) 
                remaining_indexes.remove(target_index)
            for source_index,target_index in zip(unselected_indexes,remaining_indexes): # unselected_index will be repermutated to remaining_index (from target_index) in original order.
                tmp_perm[target_index] = source_index 
            perms_buf.append(tmp_perm)
        return [tf.transpose(tensor,perm) for perm in perms_buf]

    def transform_high_dimension_inputs(self,inputs):
        """
        Transform a tensor if its shape (dimension) different from the basic model needs.
    
        Consider a N dimension input in shape [B,D1,D2,...,D{N-3},D{N-2},C]  or  [B,C,D1,D2,...,D{N-3},D{N-2}]
        Each dimension in `D1,D2,...,D{N-2}` is meaningful dimension. If a basic model only takes last 3 dimensions
        as special dimensions, i.g., vgg take [D{N-3},D{N-2},C] as [H,W,3] dimension, the `B,D1,D2,...,D{N-4}` will be broad sense batch dimension,
        which is unfair for `D1,D2,...,D{N-4}` dimensions.
            For example,
            Consider a specific vgg layer, input should be [B,H,W,3] in shape.
            `B` dimension  is batch dimension and will be preserved when calculation.
            If a tensor has more than this shape, like [B,D1,D2,D3,...,H,W,3],
            the `B,D1,D2,D3,...` dimension will be considered as general (broad sense) 
            batch dimension and preserved when calculation.
        Actually, each meaningful dimension should be treated fairly.

        A more appropriate method to deal with this poser is transforming the tensor to a list of tensor. Here are the steps in detail:
            1. consider an input in shape [B,D1,D2,...,D{N-3},D{N-2},C]  or  [B,C,D1,D2,...,D{N-3},D{N-2}]
                the meaningful_dimension_indexes is [1,2,...,N-2] or [2,3,...,N-1]
                find some information about basic model:
                    real batch_dimension_index, 0 ususally
                    channles_dimension_index, 1 or -1 ususally
                    fixed_dimension_indexes (represents the unchanged dimension index when transforming), which consists of batch_dimension_index and channles_dimension_index usually
                    target_dimension_indexes (represents the working dimension index, i.e., the target dimension index indewhen transforming), -3 and -2 usually (such as vgg's H,W dimension)
            2. if len(target_dimension_indexes)==n
                Choose n dimensions from meaningful dimensions, maintain their original relative order, transpose these selected dimensions to target dimensions.
                Each choose-transpose method will be recorded in a list.
                Exhaust all choise, we got a total list as the transforming result.
            3. The operation of step `2` can be conducted by  dimensions index.
            4. Optional step:
                With the help of broad sense batch dimension, 
                 if the result list can be stacked following real batch dimension, it will be speed up.
                Try this step.
        
        So, by the transforming, if tensor's shape different from the basic model needs, we will transform it to a list of tensor.
        Before transforming, there is a tensor and a basic model's output by this tensor.
        After transforming, there will be a list of tensor and a list of basic model's output correspondingly.
        
        If a posterior operation calculates scalars on each element of these output list and makes an average,
        the finally result will be very reasonable, treating each meaningful dimension in a balanced way.  

        A more concreate example:
            basic model is vgg16
            input shape is [7,4,5,6,3]
            self._meta_data["reperm_target_indexes"] = [-3,-2]
            self._meta_data["reperm_fixed_indexes"] = [0,-1]
            inputs_list = [tensor1,tensor2,tensor3]
            tensor1 is in shape [7,6,4,5,3]
            tensor2 is in shape [7,5,4,6,3]
            tensor3 is in shape [7,4,5,6,3]

            since inputs_list cannot be stacked,
            transforming result is just inputs_list     
        """       
        inputs_list = self._tensor_repermutation(
                        inputs,reperm_target_indexes=self._additional_meta_data["reperm_target_indexes"],
                        reperm_fixed_indexes=self._additional_meta_data["reperm_fixed_indexes"])
        try:
            new_inputs = tf.concat(inputs_list,axis=0) # if elements in inputs_list have the same shape, concat then in batch dimension for speed up
        except (ValueError,tf.errors.InvalidArgumentError):
            new_inputs = inputs_list
        return new_inputs
    def summary(self):
        self._model.summary()
    def call(self,inputs,**kwargs):
        x = self._normalize_input(inputs)
        output_buf = [[],]*len(self._feature_maps_indicators)
        for index in self._additional_meta_data["valid_layer_indexes"]:
            _layer  = self._model.get_layer(index=index)
            y = _layer(x,**kwargs)
            output_buf = self._dist_tensor_to_2D_container(tensor=y,
                        index=index,indicator=self._feature_maps_indicators,
                        container=output_buf)
            x = y
            if index >= self._fused_index:
                break
        return output_buf

class PerceptualLossExtractor(tf.keras.Model):
    """
    see: https://arxiv.org/pdf/1603.08155.pdf

    calculation perceptual loss between inputs[0](y_true) and inputs[1](y_pred)
    since perceptual loss needs a pretrained model as feature extractor
    so this class in inherited from tf.keras.Model
    the a pretrained model is managed by class FeatureMapsGetter, as basic model

    Args:
        name:name,
        model_name:model_name,
        data_format:represents the data_format of inputs[0] and inputs[1]
        transform_high_dimension: 
            If True, inputs[0] and inputs[1] will be transformed to a list of tensor when their shape different from basic model's needs
            If False, inputs[0] and inputs[1] will be maintained, meanwhile, their meaningful dimensions will not be treated fairly. 
        use_pooling:
            If False, pooling layers in basic model will be skipped when inference.
            If True, pooling layers in basic model will be used as normal when inference.
        use_feature_reco_loss: indicate whether calculate feature reconstruction loss
        use_style_reco_loss: indicate whether calculate style reconstruction loss
        feature_reco_index: for feature reconstruction loss, it indicate the layer index that will be gathered as feature maps by FeatureMapsGetter
        feature_reco_sample_weight:for feature reconstruction loss, it indicate the sample_weight of the loss between each tow feature maps of inputs[0] and inputs[1]
        style_reco_index: for style reconstruction loss, indicate the layer index that will be gathered as feature maps by FeatureMapsGetter
        style_reco_sample_weight:for style reconstruction loss, it indicate the sample_weight of the loss between each tow feature maps of inputs[0] and inputs[1]
    """
    @typechecked
    def __init__(self,
                 name:Union[None,str]=None,
                 model_name:str="vgg16",
                 data_format:str="channels_last",
                 transform_high_dimension:bool=True,
                 use_pooling:bool=True,
                 use_feature_reco_loss:bool=True,
                 use_style_reco_loss:bool=True,
                 feature_reco_index:List[int]=[5,],
                 feature_reco_sample_weight:List[Union[int,float]]=[1,],
                 style_reco_index:List[int]=[2,5,9,13],
                 style_reco_sample_weight:List[Union[int,float]]=[1,1,1,1],
                 **kwargs):
        if name is not None:
            name = name+"_"+model_name+"_perceptual_loss_extractor"
        else:
            name = model_name+"_perceptual_loss_extractor"
        if "dtype" in kwargs.keys():
            if kwargs["dtype"] is not None:
                logging.warning("""
                Setting FeatureMapsGetter's dtype to a specific dtype but not `None` may fail to meet the user's expectations. Since the actually dtype should follow 
                model's practical dtype. For numerical stability, we mandatorily set dtype to None. 
                """)
        kwargs["dtype"] = None
        super(PerceptualLossExtractor,self).__init__(name=name,**kwargs)
        
        self.data_format = data_format.lower() #  inputs' data_format, received by call()
        if self.data_format not in ["channels_first","channels_last"]:
            raise ValueError("data_format of PerceptualLoss should be 'channels_last' or 'channels_first', not {}.".format(data_format))
        self.transform_high_dimension = transform_high_dimension
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
        
        feature_maps_indicators=(tuple(feature_reco_index) if use_feature_reco_loss else tuple([]),tuple(style_reco_index) if use_style_reco_loss else tuple([]))
        self.feature_maps_getter = FeatureMapsGetter(name=name,
                                    model_name=model_name,
                                    data_format=data_format,
                                    use_pooling=use_pooling,
                                    feature_maps_indicators=feature_maps_indicators,
                                    **kwargs)
    def get_feature_maps(self,inputs,feature_reco_sample_weight,style_reco_sample_weight,**kwargs):
        """
        If inputs is a list, it means inputs has been transformed as a list of tensor, since its original shape has high_dimension that cannot treated fairly by basic model.
        So, sample_weight, given by user, should be expand simultaneously and automatically. A user dose not need to care about the expanding detail.
        
        Since posterior loss calculation step is just an average cross each feature map,
        if an inputs tensor has been transformed to a list of tensor of `n` elements, we can still gather the `n` times feature maps in a list, the result of loss 
        calculation will be equivalent, as long as sample_weight expanded correctly.
        """
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
        ## disable input shape check to speed up
        # input_shape = tf.TensorShape(input_shape)
        # assert input_shape[0]==2
        # input_shape = [None,]*(len(input_shape)-1)
        # input_shape = tuple([2]+input_shape)
        # self.input_spec = tf.keras.layers.InputSpec(shape=input_shape)  # input_spec check input shape mandatorily, so for different input shape, there should be different instance of this class
    def call(self,inputs,**kwargs):
        if self.transform_high_dimension:
            inputs_true,inputs_pred = self.feature_maps_getter.transform_high_dimension_inputs(inputs[0]),self.feature_maps_getter.transform_high_dimension_inputs(inputs[1])
        else:
            inputs_true,inputs_pred = inputs[0],inputs[1]
        kwargs["training"] = False # force un-training

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
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    feature_maps_indicators = ((1,2,4,5),())
    feature_maps_getter = FeatureMapsGetter(use_pooling=False,feature_maps_indicators=feature_maps_indicators,dtype=policy)
    x_ = tf.transpose(x_,perm=[0,3,1,2,4])
    x_ = tf.transpose(x_,perm=[0,3,1,2,4])
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


    x_  = tf.random.uniform(shape=[1,1,16,128,128]) 
    vf = PerceptualLossExtractor(model_name="vgg16",
                 data_format="channels_first",
                 transform_high_dimension=True,
                 use_pooling=True,
                 use_feature_reco_loss=True,
                 use_style_reco_loss=False,)
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


   
