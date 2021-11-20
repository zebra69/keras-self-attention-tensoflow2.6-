import tensorflow as tf

class MyAttention(tf.keras.layers.Layer):
    def __init__(self,scale=150,use_scale=False,use_bias=False,use_gamma=False, **kwargs):
        super(MyAttention, self).__init__(**kwargs)
        self.use_bias = use_bias
        self.use_scale = use_scale
        self.use_gamma = use_gamma
        self.scale = scale
    
    def build(self, input_shape):
        self.w = self.add_weight(shape=(1,input_shape[-1]),
                                 name='weight',
                                 dtype=tf.float32,
                                 initializer='ones',
                                 trainable=True)

        if self.use_bias:
            self.b = self.add_weight(shape=(input_shape[-1],),
                                     name='bias',
                                     initializer='zeros',
                                     trainable=True)

        if self.use_scale:
            self.scale = tf.cast(self.scale, tf.float32)
            self.scale = tf.sqrt(self.scale)
        
        if self.use_gamma:
            self.g = self.add_weight(shape=(1,input_shape[-1]),
                                     name='gamma',
                                     initializer='ones',
                                     trainable=True)

        super(MyAttention, self).build(input_shape)

    def call(self, x):
        e = tf.matmul(tf.keras.backend.expand_dims(x), self.w)
        if self.use_bias:
            e = tf.nn.bias_add(e, self.b)
        e = tf.math.exp(e)
        e = tf.keras.activations.softmax(e)

        if self.use_gamma:
            e = tf.matmul(self.g,e,transpose_b=True)
            a = tf.squeeze(e,[1])
            a = a*x
        else:
            a = tf.matmul(a,tf.keras.backend.expand_dims(x))
            a = tf.squeeze(a,[2])
        
        if self.use_scale:
            a/=self.scale
            
        return a

    def get_config(self):
            config = super(MyAttention, self).get_config()
            config.update({'use_bias':self.use_bias,
                           'use_scale':self.use_scale,
                           'use_gamma':self.use_gamma,
                           'scale':self.scale,})
            return config
