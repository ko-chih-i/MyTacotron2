import collections
import tensorflow as tf


class CustomDecoderOutput(
    collections.namedtuple("CustomDecoderOutput", ("rnn_output", "token_output", "sample_id"))
):
    """Output of the custom decoder."""
    pass


class CustomDecoder(tf.keras.layers.Layer):
    """Custom sampling decoder for Tacotron 2."""

    def __init__(self, cell, sampler, output_layer=None):
        """
        Initialize CustomDecoder.

        Args:
            cell: A tf.keras.layers.RNNCell instance.
            sampler: A sampler instance for training or inference.
            output_layer: An optional tf.keras.layers.Dense layer for projecting outputs.
        """
        super(CustomDecoder, self).__init__()
        self.cell = cell
        self.sampler = sampler
        self.output_layer = output_layer

    @property
    def batch_size(self):
        """Returns the batch size of the sampler."""
        return self.sampler.batch_size

    @property
    def output_size(self):
        """Defines the output size of the decoder."""
        return CustomDecoderOutput(
            rnn_output=self.cell.output_size,
            token_output=self.cell.output_size,
            sample_id=self.sampler.sample_ids_shape,
        )

    @property
    def output_dtype(self):
        """Defines the output dtype of the decoder."""
        dtype = tf.float32  # Assume RNN cell outputs are float32
        return CustomDecoderOutput(
            rnn_output=dtype, token_output=dtype, sample_id=self.sampler.sample_ids_dtype
        )

    def initialize(self, initial_state, inputs):
        """
        Initialize the decoder.

        Args:
            initial_state: Initial state for the RNN cell.
            inputs: Inputs to the sampler.

        Returns:
            Tuple of finished flags, initial inputs, and initial state.
        """
        self.initial_state = initial_state
        self.inputs = inputs
        finished, initial_inputs = self.sampler.initialize(inputs)
        return finished, initial_inputs, initial_state

    def step(self, time, inputs, state):
        """
        Perform a decoding step.

        Args:
            time: Current time step.
            inputs: Input tensors at the current step.
            state: Current state of the RNN cell.

        Returns:
            Tuple of outputs, next state, next inputs, and finished flags.
        """
        # Call the RNN cell
        rnn_output, next_state = self.cell(inputs, state)

        # Apply the output layer if it exists
        if self.output_layer is not None:
            rnn_output = self.output_layer(rnn_output)

        # Get sample ids from the sampler
        sample_ids = self.sampler.sample(time=time, outputs=rnn_output, state=next_state)

        # Get next inputs and finished flags from the sampler
        finished, next_inputs = self.sampler.next_inputs(
            time=time, outputs=rnn_output, state=next_state, sample_ids=sample_ids
        )

        # Create the output
        outputs = CustomDecoderOutput(rnn_output=rnn_output, token_output=rnn_output, sample_id=sample_ids)
        return outputs, next_state, next_inputs, finished

    def call(self, inputs, initial_state):
        """
        Implements the decoding loop.

        Args:
            inputs: Inputs to the decoder.
            initial_state: Initial state for the RNN cell.

        Returns:
            All outputs, states, and final sample ids.
        """
        finished, next_inputs, state = self.initialize(initial_state, inputs)
        time = tf.constant(0, dtype=tf.int32)
        outputs = []

        while not tf.reduce_all(finished):
            output, state, next_inputs, finished = self.step(time, next_inputs, state)
            outputs.append(output)
            time += 1

        final_outputs = tf.nest.map_structure(lambda *tensors: tf.stack(tensors), *outputs)
        return final_outputs, state

