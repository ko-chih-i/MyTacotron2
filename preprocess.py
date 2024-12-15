import argparse
import os
from multiprocessing import cpu_count

from datasets import preprocessor
from hparams import hparams
from tqdm import tqdm

import argparse
import os
from multiprocessing import cpu_count

from datasets import preprocessor
from hparams import hparams
from tqdm import tqdm


def override_hparams(hparams, hparams_str):
    """Override default hparams with user-specified values."""
    if hparams_str:
        for pair in hparams_str.split(','):
            key, value = pair.split('=')
            if key in hparams:
                try:
                    # 嘗試將值轉換為數字或布爾類型
                    value = eval(value)
                except:
                    pass
                hparams[key] = value
            else:
                raise KeyError(f"Invalid hyperparameter: {key}")


def preprocess(args, input_folders, out_dir, hparams):
    mel_dir = os.path.join(out_dir, 'mels')
    wav_dir = os.path.join(out_dir, 'audio')
    linear_dir = os.path.join(out_dir, 'linear')
    os.makedirs(mel_dir, exist_ok=True)
    os.makedirs(wav_dir, exist_ok=True)
    os.makedirs(linear_dir, exist_ok=True)
    metadata = preprocessor.build_from_path(hparams, input_folders, mel_dir, linear_dir, wav_dir, args.n_jobs, tqdm=tqdm)
    write_metadata(metadata, out_dir)


def write_metadata(metadata, out_dir):
    with open(os.path.join(out_dir, 'train.txt'), 'w', encoding='utf-8') as f:
        for m in metadata:
            f.write('|'.join([str(x) for x in m]) + '\n')
    mel_frames = sum([int(m[4]) for m in metadata])
    timesteps = sum([int(m[3]) for m in metadata])
    sr = hparams.sample_rate
    hours = timesteps / sr / 3600
    print(f'Write {len(metadata)} utterances, {mel_frames} mel frames, {timesteps} audio timesteps, ({hours:.2f} hours)')
    print(f'Max input length (text chars): {max(len(m[5]) for m in metadata)}')
    print(f'Max mel frames length: {max(int(m[4]) for m in metadata)}')
    print(f'Max audio timesteps length: {max(m[3] for m in metadata)}')


def norm_data(args):

    merge_books = (args.merge_books == 'True')

    print('Selecting data folders..')
    supported_datasets = ['LJSpeech-1.0', 'LJSpeech-1.1', 'M-AILABS']
    if args.dataset not in supported_datasets:
        raise ValueError(f'Dataset value entered {args.dataset} does not belong to supported datasets: {supported_datasets}')

    if args.dataset.startswith('LJSpeech'):
        return [os.path.join(args.base_dir, args.dataset)]

    if args.dataset == 'M-AILABS':
        supported_languages = ['en_US', 'en_UK', 'fr_FR', 'it_IT', 'de_DE', 'es_ES', 'ru_RU',
                               'uk_UK', 'pl_PL', 'nl_NL', 'pt_PT', 'fi_FI', 'se_SE', 'tr_TR', 'ar_SA']
        if args.language not in supported_languages:
            raise ValueError(f'Please enter a supported language to use from M-AILABS dataset! \n{supported_languages}')

        supported_voices = ['female', 'male', 'mix']
        if args.voice not in supported_voices:
            raise ValueError(f'Please enter a supported voice option to use from M-AILABS dataset! \n{supported_voices}')

        path = os.path.join(args.base_dir, args.language, 'by_book', args.voice)
        supported_readers = [e for e in os.listdir(path) if os.path.isdir(os.path.join(path, e))]
        if args.reader not in supported_readers:
            raise ValueError(f'Please enter a valid reader for your language and voice settings! \n{supported_readers}')

        path = os.path.join(path, args.reader)
        supported_books = [e for e in os.listdir(path) if os.path.isdir(os.path.join(path, e))]
        if merge_books:
            return [os.path.join(path, book) for book in supported_books]

        else:
            if args.book not in supported_books:
                raise ValueError(f'Please enter a valid book for your reader settings! \n{supported_books}')

            return [os.path.join(path, args.book)]


def run_preprocess(args, hparams):
    input_folders = norm_data(args)
    output_folder = os.path.join(args.base_dir, args.output)

    preprocess(args, input_folders, output_folder, hparams)


def main():
    print('Initializing preprocessing..')
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default="/content/drive/MyDrive", help='Base directory for input/output')
    parser.add_argument('--hparams', default='', help='Hyperparameter overrides as a comma-separated list of name=value pairs')
    parser.add_argument('--dataset', default='LJSpeech-1.1', help='Dataset to process')
    parser.add_argument('--language', default='en_US', help='Language of the dataset')
    parser.add_argument('--voice', default='female', help='Voice type for the dataset')
    parser.add_argument('--reader', default='mary_ann', help='Reader for the dataset')
    parser.add_argument('--merge_books', default='False', help='Merge multiple books or not')
    parser.add_argument('--book', default='northandsouth', help='Specific book to process if not merging')
    parser.add_argument('--output', default='training_data', help='Output directory for processed data')
    parser.add_argument('--n_jobs', type=int, default=cpu_count(), help='Number of parallel jobs for processing')
    args = parser.parse_args()

    # 使用新的 override_hparams 函數
    override_hparams(hparams, args.hparams)

    # 驗證 merge_books 參數值
    assert args.merge_books in ('False', 'True')

    # 執行預處理
    run_preprocess(args, hparams)


if __name__ == '__main__':
    main()
