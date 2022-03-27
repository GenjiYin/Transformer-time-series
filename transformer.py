import os
import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

"""
Transformer:
1. Encoding layer(Two blocks)

# Identity map
# Self attention
# Add and layernormalization

# Feed backward layer
# Add and layernormalization

2. Decoding layer(Three blocks)

# Identity map
# Self attention with masking
# Add and layernormalization

# React attention(Compute with encoding layer's output)
# Add and layernormalization

# Feed backward layer
# Add and layernormalization
"""

# Attention function
class Self_attention(layers.Layer):
    def __init__(self, units, heads, mask=False):
        """
        :params units: Dimension of output
        :params heads: Attention's head
        :params mask: Masking or not
        """
        super(Self_attention, self).__init__()

        self.mask = mask
        self.Wq = layers.Dense(units*heads, use_bias=False)
        self.Wk = layers.Dense(units*heads, use_bias=False)
        self.Wv = layers.Dense(units*heads, use_bias=False)

    def call(self, data):
        q  =self.Wq(data)
        k = self.Wk(data)
        v = self.Wv(data)
        weight = q@tf.transpose(k, perm=[0, 2, 1])
        shape = weight.shape
        d = tf.constant(shape[-1], dtype=tf.float32)
        d = tf.sqrt(d)
        weight = weight/d

        if self.mask:
            weight += 1000000
            one = tf.ones([shape[0], shape[1], shape[2]])
            one = tf.linalg.band_part(one, -1, 0)
            weight = weight*one - 1000000
            weight = tf.nn.softmax(weight, axis=-1)
        else:
            weight = tf.nn.softmax(weight, axis=-1)
        out = weight@v
        return out

class Feed_backward(layers.Layer):
    def __init__(self, units, length):
        """
        :params length: Length of sequence
        """
        super(Feed_backward, self).__init__()
        self.length = length
        self.feedback = [layers.Dense(units, activation=tf.nn.relu) for _ in range(length)]
    
    def call(self, data):
        output = []
        for i in range(self.length):
            df = data[:, i, :]
            out = self.feedback[i](df)
            shape = out.shape
            out = tf.reshape(out, (shape[0], 1, shape[1]))
            output.append(out)
        output = tf.concat(output, axis=1)
        return output


# Encoding layer
class Encoding(layers.Layer):
    def __init__(self, units, heads, length):
        """
        :params units: Dimension of output
        :params heads: Attention's head
        """
        super(Encoding, self).__init__()
        self.Idmap1 = layers.Dense(units*heads)
        self.Idmap2 = layers.Dense(units)
        self.attention = Self_attention(units, heads)
        self.layernorm1 = layers.LayerNormalization()
        self.layernorm2 = layers.LayerNormalization()
        self.feed = Feed_backward(units, length)
    
    def call(self, data):
        # First block
        h = self.Idmap1(data)
        out = self.attention(data)
        out = out + h

        # Second block
        h = self.Idmap2(out)
        out = self.feed(out)
        out = out + h
        return out

# Decoding
# React attention
class React_attention(layers.Layer):
    def __init__(self, units):
        super(React_attention, self).__init__()
        self.encoding_K = layers.Dense(units, use_bias=False)
        self.encoding_V = layers.Dense(units, use_bias=False)
        self.decoding_Q = layers.Dense(units, use_bias=False)
    
    def call(self, encoding, decoding):
        """
        :params encoding: Tensor from encoding layer
        :params decoding: Tensor from train's label
        """
        q = self.decoding_Q(decoding)
        k = self.encoding_K(encoding)
        v = self.encoding_V(encoding)
        score = q@tf.transpose(k, perm=[0, 2, 1])
        score = tf.nn.softmax(score, axis=2)
        return score@v

# Decoding layer
class Decoding(layers.Layer):
    def __init__(self, units, heads, length):
        super(Decoding, self).__init__()
        
        self.IDmap1 = layers.Dense(units*heads)
        self.attention = Self_attention(units, heads, mask=True)
        self.layernorm1 = layers.LayerNormalization()
        
        self.IDmap2 = layers.Dense(units)
        self.react_attention = React_attention(units)
        self.layernorm2 = layers.LayerNormalization()

        self.IDmap3 = layers.Dense(units)
        self.feed_forward = Feed_backward(units, length)
        self.layernorm3 = layers.LayerNormalization()
    
    def call(self, encoding, label_embedding):
        h1 = self.IDmap1(label_embedding)
        out = self.attention(label_embedding)
        out = out + h1
        out = self.layernorm1(out)

        h2 = self.IDmap2(out)
        out = self.react_attention(encoding, out)
        out = out + h2
        out = self.layernorm2(out)

        h3 = self.IDmap3(out)
        out = self.feed_forward(out)
        out = out + h3
        out = self.layernorm3(out)
        return out

# Transform model
class Transform(Model):
    def __init__(self, xtrain, ytrain, encoding_num, decoding_num):
        super(Transform, self).__init__()

        self.xtrain = xtrain
        self.ytrain = ytrain
        self.Decoding = [Decoding(64, 2, ytrain.shape[1]) for _ in range(decoding_num)]
        self.Encoding = Sequential([Encoding(64, 2, xtrain.shape[1]) for _ in range(encoding_num)])
    
    def call(self, xtrain, ytrain):
        out = self.Encoding(xtrain)

        label_out = ytrain
        for i in range(len(self.Decoding)):
            label_out = self.Decoding[i](out, label_out)

        return label_out

if __name__ == '__main__':
    x = tf.random.truncated_normal([100, 5, 2])
    y = tf.random.truncated_normal([100, 80, 8])
    d = Transform(x, y, 8, 8)
    print(d(x, y))