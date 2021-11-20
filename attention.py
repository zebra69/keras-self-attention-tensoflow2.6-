import tensorflow as tf

class MyAttention(tf.keras.layers.Layer):
    def __init__(self,scale=150,use_scale=False,use_bias=False,use_gamma=False, **kwargs):
        super(MyAttention, self).__init__(**kwargs)
        self.use_bias = use_bias
        self.use_scale = use_scale
        self.use_gamma = use_gamma
        self.scale = scale
    
    def build(self, input_shape):
        self.w1 = self.add_weight(shape=(1,input_shape[-1]),
                                  name='bert_weight',
                                  dtype=tf.float32,
                                  initializer='ones',
                                  trainable=True)
        self.w2 = self.add_weight(shape=(1,input_shape[-1]),
                                  name='dic_weight',
                                  dtype=tf.float32,
                                  initializer='ones',
                                  trainable=True)

        if self.use_bias:
            self.b1 = self.add_weight(shape=(input_shape[-1],),
                                      name='bert_bias',
                                      initializer='zeros',
                                      trainable=True)
            self.b2 = self.add_weight(shape=(input_shape[-1],),
                                      name='dic_bias',
                                      initializer='zeros',
                                      trainable=True)
        if self.use_scale:
            self.scale = tf.cast(self.scale, tf.float32)
            self.scale = tf.sqrt(self.scale)
        
        if self.use_gamma:
            self.g1 = self.add_weight(shape=(1,input_shape[-1]),
                                      name='bert_gamma',
                                      initializer='ones',
                                      trainable=True)
            self.g2 = self.add_weight(shape=(1,input_shape[-1]),
                                      name='dic_gamma',
                                      initializer='ones',
                                      trainable=True)

        super(MyAttention, self).build(input_shape)

    def call(self, bert, dic):
        e1 = tf.matmul(tf.keras.backend.expand_dims(bert), self.w1)
        e2 = tf.matmul(tf.keras.backend.expand_dims(dic), self.w2)
        if self.use_bias:
            e1 = tf.nn.bias_add(e1, self.b1)
            e2 = tf.nn.bias_add(e2, self.b2)
        e1,e2 = tf.math.exp(e1),tf.math.exp(e2)
        e1 = tf.keras.activations.softmax(e1)
        e2 = tf.keras.activations.softmax(e2)

        if self.use_gamma:
            e1 = tf.matmul(self.g1,e1,transpose_b=True)
            e2 = tf.matmul(self.g2,e2,transpose_b=True)
        
        a1 = e1/(e1+e2)
        a2 = e2/(e1+e2)
        
        if self.use_gamma:
            a1,a2 = tf.squeeze(a1,[1]),tf.squeeze(a2,[1])
            a1 = a1*bert
            a2 = a2*dic
        else:
            a1 = tf.matmul(a1,tf.keras.backend.expand_dims(bert))
            a2 = tf.matmul(a2,tf.keras.backend.expand_dims(dic))
            a1,a2 = tf.squeeze(a1,[2]),tf.squeeze(a2,[2])
        
        if self.use_scale:
            a1/=self.scale
            a2/=self.scale
            
        return a1+a2

    def get_config(self):
            config = super(MyAttention, self).get_config()
            config.update({'use_bias':self.use_bias,
                           'use_scale':self.use_scale,
                           'use_gamma':self.use_gamma,
                           'scale':self.scale,})
            return config
