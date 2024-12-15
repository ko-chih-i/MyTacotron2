"""Attention file for location based attention (compatible with tensorflow attention wrapper)"""
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv1D, AdditiveAttention
from tensorflow.keras.initializers import GlorotUniform


def _compute_attention(attention_mechanism, cell_output, attention_state, attention_layer, prev_max_attentions):
    """Computes the attention and alignments for a given attention_mechanism."""
    alignments, next_attention_state, max_attentions = attention_mechanism(
        cell_output, state=attention_state, prev_max_attentions=prev_max_attentions)

    # Reshape from [batch_size, memory_time] to [batch_size, 1, memory_time]
    expanded_alignments = tf.expand_dims(alignments, 1)
    # Compute context vector
    context = tf.matmul(expanded_alignments, attention_mechanism.values)
    context = tf.squeeze(context, [1])

    if attention_layer is not None:
        attention = attention_layer(tf.concat([cell_output, context], axis=1))
    else:
        attention = context

    return attention, alignments, next_attention_state, max_attentions


def _location_sensitive_score(W_query, W_fil, W_keys):
    """Implements Bahdanau-style (cumulative) scoring function."""
    dtype = W_query.dtype
    num_units = W_keys.shape[-1]

    v_a = tf.Variable(GlorotUniform()(shape=[num_units]), name='attention_variable_projection', dtype=dtype)
    b_a = tf.Variable(tf.zeros([num_units]), name='attention_bias', dtype=dtype)

    return tf.reduce_sum(v_a * tf.tanh(W_keys + W_query + W_fil + b_a), axis=2)


def _smoothing_normalization(e):
    """Applies a smoothing normalization function instead of softmax."""
    return tf.nn.sigmoid(e) / tf.reduce_sum(tf.nn.sigmoid(e), axis=-1, keepdims=True)


class LocationSensitiveAttention(tf.keras.layers.Layer):
    """Implements Bahdanau-style (cumulative) scoring function."""

    def __init__(self, num_units, memory, hparams, is_training, mask_encoder=True, memory_sequence_length=None,
                 smoothing=False, cumulate_weights=True, name='LocationSensitiveAttention'):
        super(LocationSensitiveAttention, self).__init__(name=name)

        self.num_units = num_units
        self.memory = memory
        self.memory_sequence_length = memory_sequence_length if mask_encoder else None
        self.cumulate_weights = cumulate_weights
        self.synthesis_constraint = hparams.synthesis_constraint and not is_training
        self.attention_win_size = tf.convert_to_tensor(hparams.attention_win_size, dtype=tf.int32)
        self.constraint_type = hparams.synthesis_constraint_type

        # Create normalization function
        self.normalization_function = _smoothing_normalization if smoothing else None

        # Layers for location-based attention
        self.location_convolution = Conv1D(filters=hparams.attention_filters,
                                           kernel_size=hparams.attention_kernel, padding='same', use_bias=True,
                                           bias_initializer=tf.zeros_initializer(), name='location_features_convolution')
        self.location_layer = Dense(units=num_units, use_bias=False, name='location_features_layer')

    def call(self, query, state, prev_max_attentions):
        """Score the query based on the keys and values."""
        previous_alignments = state
        processed_query = tf.expand_dims(query, 1)

        expanded_alignments = tf.expand_dims(previous_alignments, axis=2)
        location_features = self.location_convolution(expanded_alignments)
        processed_location_features = self.location_layer(location_features)

        energy = _location_sensitive_score(processed_query, processed_location_features, self.memory)

        if self.synthesis_constraint:
            Tx = tf.shape(energy)[-1]
            if self.constraint_type == 'monotonic':
                key_masks = tf.sequence_mask(prev_max_attentions, Tx)
                reverse_masks = tf.sequence_mask(Tx - self.attention_win_size - prev_max_attentions, Tx)[:, ::-1]
            else:
                assert self.constraint_type == 'window'
                key_masks = tf.sequence_mask(prev_max_attentions - (self.attention_win_size // 2 +
                                                                    (self.attention_win_size % 2 != 0)), Tx)
                reverse_masks = tf.sequence_mask(Tx - (self.attention_win_size // 2) - prev_max_attentions, Tx)[:, ::-1]

            masks = tf.logical_or(key_masks, reverse_masks)
            paddings = tf.ones_like(energy) * (-2 ** 32 + 1)
            energy = tf.where(tf.equal(masks, False), energy, paddings)

        alignments = self.normalization_function(energy) if self.normalization_function else tf.nn.softmax(energy)
        max_attentions = tf.argmax(alignments, axis=-1, output_type=tf.int32)

        if self.cumulate_weights:
            next_state = alignments + previous_alignments
        else:
            next_state = alignments

        return alignments, next_state, max_attentions
