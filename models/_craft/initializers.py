import tensorflow as tf
__all__ = [ 
    "initializer_slect",
]
def initializer_slect(initializer_name,*args,**kwargs):
    if initializer_name == "ones":
        return tf.initializers.Ones()
    elif initializer_name == "zeros":
        return tf.initializers.Zeros()
    elif initializer_name == "glorot_normal":
        return tf.initializers.GlorotNormal(*args,**kwargs)
    elif initializer_name == "glorot_uniform":
        return tf.initializers.GlorotUniform(*args,**kwargs)
    elif initializer_name == "random_normal":
        return tf.initializers.RandomNormal(*args,**kwargs)
    elif initializer_name == "random_uniform":
        return tf.initializers.RandomUniform(*args,**kwargs)
    elif initializer_name == "truncated_normal":
        return tf.initializers.TruncatedNormal(*args,**kwargs)
    else:
        raise ValueError("Unsupported initializers: "+initializer_name)