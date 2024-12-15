import os
import threading
import time
import traceback

import numpy as np
import tensorflow as tf
from infolog import log
from sklearn.model_selection import train_test_split
from tacotron.utils.text import text_to_sequence

_batches_per_group = 64

class Feeder:
    """
    Feeds batches of data into queue on a background thread.
    """

    def __init__(self, coordinator, metadata_filename, hparams):
        super(Feeder, self).__init__()
        self._coord = coordinator
        self._hparams = hparams
        self._cleaner_names = [x.strip() for x in hparams.cleaners.split(',')]
        self._train_offset = 0
        self._test_offset = 0

        # Load metadata
        self._mel_dir = os.path.join(os.path.dirname(metadata_filename), 'mels')
        self._linear_dir = os.path.join(os.path.dirname(metadata_filename), 'linear')
        with open(metadata_filename, encoding='utf-8') as f:
            self._metadata = [line.strip().split('|') for line in f]
            frame_shift_ms = hparams.hop_size / hparams.sample_rate
            hours = sum([int(x[4]) for x in self._metadata if len(x) > 4]) * frame_shift_ms / 3600
            log(f'Loaded metadata for {len(self._metadata)} examples ({hours:.2f} hours)')

        # Train test split
        if hparams.tacotron_test_size is None:
            assert hparams.tacotron_test_batches is not None

        test_size = (hparams.tacotron_test_size if hparams.tacotron_test_size is not None
                     else hparams.tacotron_test_batches * hparams.tacotron_batch_size)
        indices = np.arange(len(self._metadata))
        train_indices, test_indices = train_test_split(indices,
                                                        test_size=test_size,
                                                        random_state=hparams.tacotron_data_random_state)

        # Make sure test_indices is a multiple of batch_size else round down
        len_test_indices = self._round_down(len(test_indices), hparams.tacotron_batch_size)
        extra_test = test_indices[len_test_indices:]
        test_indices = test_indices[:len_test_indices]
        train_indices = np.concatenate([train_indices, extra_test])

        self._train_meta = list(np.array(self._metadata)[train_indices])
        self._test_meta = list(np.array(self._metadata)[test_indices])

        self.test_steps = len(self._test_meta) // hparams.tacotron_batch_size

        if hparams.tacotron_test_size is None:
            assert hparams.tacotron_test_batches == self.test_steps

        # Pad input sequences with the <pad_token> 0 ( _ )
        self._pad = 0
        if hparams.symmetric_mels:
            self._target_pad = -hparams.max_abs_value
        else:
            self._target_pad = 0.
        self._token_pad = 1.

        # Create placeholders or inputs using tf.keras.Input for TensorFlow 2.x
        self.inputs = tf.keras.Input(shape=(None, None), dtype=tf.int32, name='inputs')
        self.input_lengths = tf.keras.Input(shape=(None,), dtype=tf.int32, name='input_lengths')
        self.mel_targets = tf.keras.Input(shape=(None, None, hparams.num_mels), dtype=tf.float32, name='mel_targets')
        self.token_targets = tf.keras.Input(shape=(None, None), dtype=tf.float32, name='token_targets')
        self.linear_targets = tf.keras.Input(shape=(None, None, hparams.num_freq), dtype=tf.float32, name='linear_targets')
        self.targets_lengths = tf.keras.Input(shape=(None,), dtype=tf.int32, name='targets_lengths')
        self.split_infos = tf.keras.Input(shape=(hparams.tacotron_num_gpus, None), dtype=tf.int32, name='split_infos')

    def start_threads(self, session):
        self._session = session
        train_thread = threading.Thread(name='train_background', target=self._enqueue_next_train_group)
        train_thread.daemon = True
        train_thread.start()

        test_thread = threading.Thread(name='test_background', target=self._enqueue_next_test_group)
        test_thread.daemon = True
        test_thread.start()

    def _enqueue_next_train_group(self):
        while not self._coord.should_stop():
            # Implement enqueue logic here
            pass

    def _enqueue_next_test_group(self):
        while not self._coord.should_stop():
            # Implement enqueue logic here
            pass

    def _round_down(self, x, multiple):
        remainder = x % multiple
        return x if remainder == 0 else x - remainder


