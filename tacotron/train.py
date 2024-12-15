import argparse
import os
import time
from datetime import datetime
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import infolog
import numpy as np
import tensorflow as tf
from datasets import audio
from hparams import hparams
from tacotron.feeder import Feeder
from tacotron.models import create_model

log = infolog.log


def time_string():
    return datetime.now().strftime('%Y-%m-%d %H:%M')


def ensure_hparams(hparams):
    """確認必要的超參數存在，否則拋出異常"""
    required_params = [
        "tacotron_train_steps", "checkpoint_interval", "summary_interval",
        "eval_interval", "tacotron_random_seed"
    ]
    for param in required_params:
        if param not in hparams:
            raise ValueError(f"缺少必要的 hparam: {param}")


def create_dirs(log_dir, hparams):
    """創建訓練所需的目錄"""
    dirs = {
        "save_dir": os.path.join(log_dir, 'taco_pretrained'),
        "plot_dir": os.path.join(log_dir, 'plots'),
        "wav_dir": os.path.join(log_dir, 'wavs'),
        "mel_dir": os.path.join(log_dir, 'mel-spectrograms'),
        "eval_dir": os.path.join(log_dir, 'eval-dir'),
        "tensorboard_dir": os.path.join(log_dir, 'tacotron_events'),
        "meta_folder": os.path.join(log_dir, 'metas'),
    }
    if hparams["predict_linear"]:
        dirs["linear_dir"] = os.path.join(log_dir, 'linear-spectrograms')
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


def train(log_dir, args, hparams):
    ensure_hparams(hparams)  # 檢查 hparams
    dirs = create_dirs(log_dir, hparams)  # 創建目錄
    checkpoint_path = os.path.join(dirs["save_dir"], 'tacotron_model.ckpt')

    log(f"Checkpoint path: {checkpoint_path}")
    log("Hyperparameters:")
    for key, value in hparams.items():
        log(f"  {key}: {value}")

    # 設定隨機種子
    tf.random.set_seed(hparams["tacotron_random_seed"])

    # 設置 GPU 設置
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    # 創建模型、資料和優化器
    coord = tf.train.Coordinator()
    feeder = Feeder(coord, args.input_path, hparams)
    global_step = tf.Variable(0, name="global_step", trainable=False)

    model = create_model(args.model, hparams)
    model.initialize(feeder.inputs, feeder.input_lengths, feeder.mel_targets, feeder.token_targets, 
                     global_step=global_step, is_training=True)
    model.add_loss()
    model.add_optimizer(global_step)

    # TensorBoard 和 Checkpoint
    summary_writer = tf.summary.create_file_writer(dirs["tensorboard_dir"])
    checkpoint = tf.train.Checkpoint(model=model, optimizer=model.optimizer)

    # 訓練循環
    for step in range(hparams["tacotron_train_steps"]):
        start_time = time.time()
        loss, _ = model.train_step()
        duration = time.time() - start_time

        log(f"Step {step}, Loss: {loss:.5f}, Time: {duration:.3f} sec")

        if step % args.checkpoint_interval == 0:
            checkpoint.save(checkpoint_path)
            log(f"Checkpoint saved at step {step}")
        
        with summary_writer.as_default():
            tf.summary.scalar("loss", loss, step=step)

    log("Training complete.")
    return dirs["save_dir"]


def tacotron_train(args, log_dir, hparams):
    return train(log_dir, args, hparams)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", default="./", help="Base directory for output")
    parser.add_argument("--input_path", default="/content/drive/MyDrive/training_data/train.txt"
, help="Path to input data")
    parser.add_argument("--model", default="Tacotron", help="Model to use")
    parser.add_argument("--checkpoint_interval", type=int, default=1000, help="Steps between checkpoints")
    parser.add_argument("--restore", action="store_true", help="Restore from checkpoint")
    args = parser.parse_args()

    log_dir = os.path.join(args.base_dir, "logs")
    tacotron_train(args, log_dir, hparams)
