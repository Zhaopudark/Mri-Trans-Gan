import math
import tensorflow as tf

def create_model(input_shape, filter_size, filters, strides, dilation_rate, padding):
    '''
        Just a simple creator function to build up a simple convolutional model with exact one layer.
    '''
    model = tf.keras.Sequential()
    #model.add(tf.keras.layers.Conv2D(2, 3, dilation_rate=dilation_rate, strides=strides, activation='relu', input_shape=(128, 128, 3)))
    model.add(
        tf.keras.layers.SeparableConv2D(
            filters = filters, 
            kernel_size = filter_size, 
            dilation_rate = dilation_rate, 
            strides = strides, 
            activation = 'relu', 
            input_shape = input_shape,
            padding = padding
        )
    )
    sgd = tf.optimizers.SGD(lr=0.1, momentum=0.9)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model



def calcShape(input_shape, filter_size, filters, strides, dilation_rate, padding):
    '''
        A simple calcShape function. Based on the provided formular given from:
        https://www.tensorflow.org/api_docs/python/tf/nn#notes_on_padding_2
    '''
    result_shape = [None]
    
    for i in range(len(input_shape)-1):
        output_shape = None

        if padding == "SAME": 
            output_shape = math.ceil(input_shape[i]/strides)
        elif padding == "VALID": 
            output_shape = math.ceil((input_shape[i]-(filter_size-1)*dilation_rate)/strides)
        
        result_shape.append(output_shape)
    result_shape.append(filters)

    return tuple(result_shape)


def compareShapes(test_no, test_params):
    '''
        Our testing function. Creates the model, extracts the convolutional layers 
        output shape and compares it against our calculated shape.
    '''
    model = create_model(**test_params)
    conv_output_shape = model.get_layer(index=0).output_shape
    tf.print(model.get_layer(index=0).depthwise_kernel.shape)
    tf.print(model.get_layer(index=0).pointwise_kernel.shape)
    # model.summary()
    calculated_shape = calcShape(**test_params)
    passed = conv_output_shape == calculated_shape

    print("")
    print('Test #{0}: Passed: {1}'.format(test_no, passed))
    print("conv_output_shape", conv_output_shape)
    print("calculated_shape", calculated_shape)
# Test 1
test_params = {
    "input_shape": (128, 128, 3),
    "filter_size": 3,
    "dilation_rate": 1,
    "strides": 3,
    "padding": "VALID",
    "filters": 200
}
compareShapes(1, test_params)

# Test 2
test_params = {
    "input_shape": (256, 256, 3),
    "filter_size": 5,
    "dilation_rate": 1,
    "strides": 5,
    "padding": "VALID",
    "filters": 200
}
compareShapes(2, test_params)

# Test 3
test_params = {
    "input_shape": (256, 256, 3),
    "filter_size": 5,
    "dilation_rate": 5,
    "strides": 3,
    "padding": "VALID",
    "filters": 200
}
compareShapes(3, test_params)


# Test 4
test_params = {
    "input_shape": (256, 256, 3),
    "filter_size": 5,
    "dilation_rate": 5,
    "strides": 7,
    "padding": "VALID",
    "filters": 200
}
compareShapes(4, test_params)


