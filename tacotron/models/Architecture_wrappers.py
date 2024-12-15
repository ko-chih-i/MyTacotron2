import collections
import tensorflow as tf
from tacotron.models.attention import _compute_attention
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops, check_ops, tensor_array_ops
from tensorflow.keras.layers import Layer

class TacotronEncoderCell(Layer):
    """Tacotron 2 Encoder Cell
    Passes inputs through a stack of convolutional layers then through a bidirectional LSTM
    layer to predict the hidden representation vector (or memory).
    """

    def __init__(self, convolutional_layers, lstm_layer):
        """Initialize encoder parameters.

        Args:
            convolutional_layers: Encoder convolutional block class
            lstm_layer: encoder bidirectional LSTM layer class
        """
        super(TacotronEncoderCell, self).__init__()
        self._convolutions = convolutional_layers
        self._cell = lstm_layer

    def call(self, inputs, input_lengths=None):
        # Pass input sequence through a stack of convolutional layers
        conv_output = self._convolutions(inputs)

        # Extract hidden representation from encoder LSTM cells
        hidden_representation = self._cell(conv_output, input_lengths)

        # For shape visualization
        self.conv_output_shape = conv_output.shape
        return hidden_representation


class TacotronDecoderCellState(
    collections.namedtuple(
        "TacotronDecoderCellState",
        ("cell_state", "attention", "time", "alignments", "alignment_history", "max_attentions"),
    )
):
    """`namedtuple` storing the state of a `TacotronDecoderCell`.

    Contains:
      - `cell_state`: The state of the wrapped `RNNCell` at the previous time step.
      - `attention`: The attention emitted at the previous time step.
      - `time`: int32 scalar containing the current time step.
      - `alignments`: A single or tuple of `Tensor`(s) containing the alignments
         emitted at the previous time step for each attention mechanism.
      - `alignment_history`: a single or tuple of `TensorArray`(s)
         containing alignment matrices from all time steps for each attention mechanism.
         Call `stack()` on each to convert to a `Tensor`.
    """

    def replace(self, **kwargs):
        """Clones the current state while overwriting components provided by kwargs."""
        return super(TacotronDecoderCellState, self)._replace(**kwargs)


class TacotronDecoderCell(Layer):
    """Tacotron 2 Decoder Cell."""

    def __init__(self, prenet, attention_mechanism, rnn_cell, frame_projection, stop_projection):
        """Initialize decoder parameters.

        Args:
            prenet: A tensorflow fully connected layer acting as the decoder pre-net
            attention_mechanism: A _BaseAttentionMechanism instance, useful to learn encoder-decoder alignments
            rnn_cell: Instance of RNNCell, main body of the decoder
            frame_projection: tensorflow fully connected layer with r * num_mels output units
            stop_projection: tensorflow fully connected layer, expected to project to a scalar and through a sigmoid activation
        """
        super(TacotronDecoderCell, self).__init__()
        self._prenet = prenet
        self._attention_mechanism = attention_mechanism
        self._cell = rnn_cell
        self._frame_projection = frame_projection
        self._stop_projection = stop_projection

    def call(self, inputs, state):
        # Information bottleneck (essential for learning attention)
        prenet_output = self._prenet(inputs)

        # Concat context vector and prenet output to form LSTM cells input (input feeding)
        LSTM_input = tf.concat([prenet_output, state.attention], axis=-1)

        # Unidirectional LSTM layers
        LSTM_output, next_cell_state = self._cell(LSTM_input, state.cell_state)

        # Compute the attention (context) vector and alignments
        previous_alignments = state.alignments
        previous_alignment_history = state.alignment_history
        context_vector, alignments, cumulated_alignments, max_attentions = _compute_attention(
            self._attention_mechanism,
            LSTM_output,
            previous_alignments,
            attention_layer=None,
            prev_max_attentions=state.max_attentions,
        )

        # Concat LSTM outputs and context vector to form projections inputs
        projections_input = tf.concat([LSTM_output, context_vector], axis=-1)

        # Compute predicted frames and predicted <stop_token>
        cell_outputs = self._frame_projection(projections_input)
        stop_tokens = self._stop_projection(projections_input)

        # Save alignment history
        alignment_history = previous_alignment_history.write(state.time, alignments)

        # Prepare next decoder state
        next_state = TacotronDecoderCellState(
            time=state.time + 1,
            cell_state=next_cell_state,
            attention=context_vector,
            alignments=cumulated_alignments,
            alignment_history=alignment_history,
            max_attentions=max_attentions,
        )

        return (cell_outputs, stop_tokens), next_state

