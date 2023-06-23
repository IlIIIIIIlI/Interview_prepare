import os
from functools import lru_cache
from typing import Union

import torch
import torchaudio
from huggingface_hub import hf_hub_download

os.system(
    "cp -v /home/user/.local/lib/python3.8/site-packages/k2/lib/*.so /home/user/.local/lib/python3.8/site-packages/sherpa/lib/"
)

import k2  # noqa
import sherpa
import sherpa_onnx
import numpy as np
from typing import Tuple
import wave

sample_rate = 16000


def read_wave(wave_filename: str) -> Tuple[np.ndarray, int]:
    """
    Args:
      wave_filename:
        Path to a wave file. It should be single channel and each sample should
        be 16-bit. Its sample rate does not need to be 16kHz.
    Returns:
      Return a tuple containing:
       - A 1-D array of dtype np.float32 containing the samples, which are
       normalized to the range [-1, 1].
       - sample rate of the wave file
    """

    with wave.open(wave_filename) as f:
        assert f.getnchannels() == 1, f.getnchannels()
        assert f.getsampwidth() == 2, f.getsampwidth()  # it is in bytes
        num_samples = f.getnframes()
        samples = f.readframes(num_samples)
        samples_int16 = np.frombuffer(samples, dtype=np.int16)
        samples_float32 = samples_int16.astype(np.float32)

        samples_float32 = samples_float32 / 32768
        return samples_float32, f.getframerate()


def decode_offline_recognizer(
    recognizer: sherpa.OfflineRecognizer,
    filename: str,
) -> str:
    s = recognizer.create_stream()

    s.accept_wave_file(filename)
    recognizer.decode_stream(s)

    text = s.result.text.strip()
    return text.lower()


def decode_online_recognizer(
    recognizer: sherpa.OnlineRecognizer,
    filename: str,
) -> str:
    samples, actual_sample_rate = torchaudio.load(filename)
    assert sample_rate == actual_sample_rate, (
        sample_rate,
        actual_sample_rate,
    )
    samples = samples[0].contiguous()

    s = recognizer.create_stream()

    tail_padding = torch.zeros(int(sample_rate * 0.3), dtype=torch.float32)
    s.accept_waveform(sample_rate, samples)
    s.accept_waveform(sample_rate, tail_padding)
    s.input_finished()

    while recognizer.is_ready(s):
        recognizer.decode_stream(s)

    text = recognizer.get_result(s).text
    return text.strip().lower()


def decode_offline_recognizer_sherpa_onnx(
    recognizer: sherpa_onnx.OfflineRecognizer,
    filename: str,
) -> str:
    s = recognizer.create_stream()
    samples, sample_rate = read_wave(filename)
    s.accept_waveform(sample_rate, samples)
    recognizer.decode_stream(s)

    return s.result.text.lower()


def decode_online_recognizer_sherpa_onnx(
    recognizer: sherpa_onnx.OnlineRecognizer,
    filename: str,
) -> str:
    s = recognizer.create_stream()
    samples, sample_rate = read_wave(filename)
    s.accept_waveform(sample_rate, samples)

    tail_paddings = np.zeros(int(0.3 * sample_rate), dtype=np.float32)
    s.accept_waveform(sample_rate, tail_paddings)
    s.input_finished()

    while recognizer.is_ready(s):
        recognizer.decode_stream(s)

    return recognizer.get_result(s).lower()


def decode(
    recognizer: Union[
        sherpa.OfflineRecognizer,
        sherpa.OnlineRecognizer,
        sherpa_onnx.OfflineRecognizer,
        sherpa_onnx.OnlineRecognizer,
    ],
    filename: str,
) -> str:
    if isinstance(recognizer, sherpa.OfflineRecognizer):
        return decode_offline_recognizer(recognizer, filename)
    elif isinstance(recognizer, sherpa.OnlineRecognizer):
        return decode_online_recognizer(recognizer, filename)
    elif isinstance(recognizer, sherpa_onnx.OfflineRecognizer):
        return decode_offline_recognizer_sherpa_onnx(recognizer, filename)
    elif isinstance(recognizer, sherpa_onnx.OnlineRecognizer):
        return decode_online_recognizer_sherpa_onnx(recognizer, filename)
    else:
        raise ValueError(f"Unknown recognizer type {type(recognizer)}")


@lru_cache(maxsize=30)
def get_pretrained_model(
    repo_id: str,
    decoding_method: str,
    num_active_paths: int,
) -> Union[sherpa.OfflineRecognizer, sherpa.OnlineRecognizer]:
    if repo_id in chinese_models:
        return chinese_models[repo_id](
            repo_id, decoding_method=decoding_method, num_active_paths=num_active_paths
        )
    elif repo_id in english_models:
        return english_models[repo_id](
            repo_id, decoding_method=decoding_method, num_active_paths=num_active_paths
        )
    elif repo_id in chinese_english_mixed_models:
        return chinese_english_mixed_models[repo_id](
            repo_id, decoding_method=decoding_method, num_active_paths=num_active_paths
        )
    elif repo_id in tibetan_models:
        return tibetan_models[repo_id](
            repo_id, decoding_method=decoding_method, num_active_paths=num_active_paths
        )
    elif repo_id in arabic_models:
        return arabic_models[repo_id](
            repo_id, decoding_method=decoding_method, num_active_paths=num_active_paths
        )
    elif repo_id in german_models:
        return german_models[repo_id](
            repo_id, decoding_method=decoding_method, num_active_paths=num_active_paths
        )
    elif repo_id in french_models:
        return french_models[repo_id](
            repo_id, decoding_method=decoding_method, num_active_paths=num_active_paths
        )
    elif repo_id in japanese_models:
        return japanese_models[repo_id](
            repo_id, decoding_method=decoding_method, num_active_paths=num_active_paths
        )
    else:
        raise ValueError(f"Unsupported repo_id: {repo_id}")


def _get_nn_model_filename(
    repo_id: str,
    filename: str,
    subfolder: str = "exp",
) -> str:
    nn_model_filename = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        subfolder=subfolder,
    )
    return nn_model_filename


def _get_bpe_model_filename(
    repo_id: str,
    filename: str = "bpe.model",
    subfolder: str = "data/lang_bpe_500",
) -> str:
    bpe_model_filename = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        subfolder=subfolder,
    )
    return bpe_model_filename


def _get_token_filename(
    repo_id: str,
    filename: str = "tokens.txt",
    subfolder: str = "data/lang_char",
) -> str:
    token_filename = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        subfolder=subfolder,
    )
    return token_filename


@lru_cache(maxsize=10)
def _get_aishell2_pretrained_model(
    repo_id: str,
    decoding_method: str,
    num_active_paths: int,
) -> sherpa.OfflineRecognizer:
    assert repo_id in [
        # context-size 1
        "yuekai/icefall-asr-aishell2-pruned-transducer-stateless5-A-2022-07-12",  # noqa
        # context-size 2
        "yuekai/icefall-asr-aishell2-pruned-transducer-stateless5-B-2022-07-12",  # noqa
    ], repo_id

    nn_model = _get_nn_model_filename(
        repo_id=repo_id,
        filename="cpu_jit.pt",
    )
    tokens = _get_token_filename(repo_id=repo_id)

    feat_config = sherpa.FeatureConfig()
    feat_config.fbank_opts.frame_opts.samp_freq = sample_rate
    feat_config.fbank_opts.mel_opts.num_bins = 80
    feat_config.fbank_opts.frame_opts.dither = 0

    config = sherpa.OfflineRecognizerConfig(
        nn_model=nn_model,
        tokens=tokens,
        use_gpu=False,
        feat_config=feat_config,
        decoding_method=decoding_method,
        num_active_paths=num_active_paths,
    )

    recognizer = sherpa.OfflineRecognizer(config)

    return recognizer


@lru_cache(maxsize=10)
def _get_gigaspeech_pre_trained_model(
    repo_id: str,
    decoding_method: str,
    num_active_paths: int,
) -> sherpa.OfflineRecognizer:
    assert repo_id in [
        "wgb14/icefall-asr-gigaspeech-pruned-transducer-stateless2",
    ], repo_id

    nn_model = _get_nn_model_filename(
        repo_id=repo_id,
        filename="cpu_jit-iter-3488000-avg-20.pt",
    )
    tokens = "./giga-tokens.txt"

    feat_config = sherpa.FeatureConfig()
    feat_config.fbank_opts.frame_opts.samp_freq = sample_rate
    feat_config.fbank_opts.mel_opts.num_bins = 80
    feat_config.fbank_opts.frame_opts.dither = 0

    config = sherpa.OfflineRecognizerConfig(
        nn_model=nn_model,
        tokens=tokens,
        use_gpu=False,
        feat_config=feat_config,
        decoding_method=decoding_method,
        num_active_paths=num_active_paths,
    )

    recognizer = sherpa.OfflineRecognizer(config)

    return recognizer


@lru_cache(maxsize=10)
def _get_english_model(
    repo_id: str,
    decoding_method: str,
    num_active_paths: int,
) -> sherpa.OfflineRecognizer:
    assert repo_id in [
        "WeijiZhuang/icefall-asr-librispeech-pruned-transducer-stateless8-2022-12-02",  # noqa
        "yfyeung/icefall-asr-multidataset-pruned_transducer_stateless7-2023-05-04",  # noqa
        "yfyeung/icefall-asr-finetune-mux-pruned_transducer_stateless7-2023-05-19",  # noqa
        "csukuangfj/icefall-asr-librispeech-pruned-transducer-stateless3-2022-05-13",  # noqa
        "csukuangfj/icefall-asr-librispeech-pruned-transducer-stateless7-2022-11-11",  # noqa
        "csukuangfj/icefall-asr-librispeech-pruned-transducer-stateless8-2022-11-14",  # noqa
        "Zengwei/icefall-asr-librispeech-zipformer-large-2023-05-16",  # noqa
        "Zengwei/icefall-asr-librispeech-zipformer-2023-05-15",  # noqa
        "Zengwei/icefall-asr-librispeech-zipformer-small-2023-05-16",  # noqa
        "videodanchik/icefall-asr-tedlium3-conformer-ctc2",
        "pkufool/icefall_asr_librispeech_conformer_ctc",
        "WayneWiser/icefall-asr-librispeech-conformer-ctc2-jit-bpe-500-2022-07-21",
    ], repo_id

    filename = "cpu_jit.pt"
    if (
        repo_id
        == "csukuangfj/icefall-asr-librispeech-pruned-transducer-stateless7-2022-11-11"
    ):
        filename = "cpu_jit-torch-1.10.0.pt"

    if (
        repo_id
        == "WeijiZhuang/icefall-asr-librispeech-pruned-transducer-stateless8-2022-12-02"
    ):
        filename = "cpu_jit-torch-1.10.pt"

    if (
        repo_id
        == "yfyeung/icefall-asr-multidataset-pruned_transducer_stateless7-2023-05-04"
    ):
        filename = "cpu_jit-epoch-30-avg-4.pt"

    if (
        repo_id
        == "yfyeung/icefall-asr-finetune-mux-pruned_transducer_stateless7-2023-05-19"
    ):
        filename = "cpu_jit-epoch-20-avg-5.pt"

    if repo_id in (
        "Zengwei/icefall-asr-librispeech-zipformer-large-2023-05-16",
        "Zengwei/icefall-asr-librispeech-zipformer-2023-05-15",
        "Zengwei/icefall-asr-librispeech-zipformer-small-2023-05-16",
    ):
        filename = "jit_script.pt"

    nn_model = _get_nn_model_filename(
        repo_id=repo_id,
        filename=filename,
    )
    subfolder = "data/lang_bpe_500"

    if repo_id in (
        "videodanchik/icefall-asr-tedlium3-conformer-ctc2",
        "pkufool/icefall_asr_librispeech_conformer_ctc",
    ):
        subfolder = "data/lang_bpe"

    tokens = _get_token_filename(repo_id=repo_id, subfolder=subfolder)

    feat_config = sherpa.FeatureConfig()
    feat_config.fbank_opts.frame_opts.samp_freq = sample_rate
    feat_config.fbank_opts.mel_opts.num_bins = 80
    feat_config.fbank_opts.frame_opts.dither = 0

    config = sherpa.OfflineRecognizerConfig(
        nn_model=nn_model,
        tokens=tokens,
        use_gpu=False,
        feat_config=feat_config,
        decoding_method=decoding_method,
        num_active_paths=num_active_paths,
    )

    recognizer = sherpa.OfflineRecognizer(config)

    return recognizer


@lru_cache(maxsize=10)
def _get_wenetspeech_pre_trained_model(
    repo_id: str,
    decoding_method: str,
    num_active_paths: int,
):
    assert repo_id in [
        "luomingshuang/icefall_asr_wenetspeech_pruned_transducer_stateless2",
    ], repo_id

    nn_model = _get_nn_model_filename(
        repo_id=repo_id,
        filename="cpu_jit_epoch_10_avg_2_torch_1.7.1.pt",
    )
    tokens = _get_token_filename(repo_id=repo_id)

    feat_config = sherpa.FeatureConfig()
    feat_config.fbank_opts.frame_opts.samp_freq = sample_rate
    feat_config.fbank_opts.mel_opts.num_bins = 80
    feat_config.fbank_opts.frame_opts.dither = 0

    config = sherpa.OfflineRecognizerConfig(
        nn_model=nn_model,
        tokens=tokens,
        use_gpu=False,
        feat_config=feat_config,
        decoding_method=decoding_method,
        num_active_paths=num_active_paths,
    )

    recognizer = sherpa.OfflineRecognizer(config)

    return recognizer


@lru_cache(maxsize=10)
def _get_chinese_english_mixed_model(
    repo_id: str,
    decoding_method: str,
    num_active_paths: int,
):
    assert repo_id in [
        "luomingshuang/icefall_asr_tal-csasr_pruned_transducer_stateless5",
        "ptrnull/icefall-asr-conv-emformer-transducer-stateless2-zh",
    ], repo_id

    if repo_id == "luomingshuang/icefall_asr_tal-csasr_pruned_transducer_stateless5":
        filename = "cpu_jit.pt"
        subfolder = "data/lang_char"
    elif repo_id == "ptrnull/icefall-asr-conv-emformer-transducer-stateless2-zh":
        filename = "cpu_jit-epoch-11-avg-1.pt"
        subfolder = "data/lang_char_bpe"

    nn_model = _get_nn_model_filename(
        repo_id=repo_id,
        filename=filename,
    )
    tokens = _get_token_filename(repo_id=repo_id, subfolder=subfolder)

    feat_config = sherpa.FeatureConfig()
    feat_config.fbank_opts.frame_opts.samp_freq = sample_rate
    feat_config.fbank_opts.mel_opts.num_bins = 80
    feat_config.fbank_opts.frame_opts.dither = 0

    config = sherpa.OfflineRecognizerConfig(
        nn_model=nn_model,
        tokens=tokens,
        use_gpu=False,
        feat_config=feat_config,
        decoding_method=decoding_method,
        num_active_paths=num_active_paths,
    )

    recognizer = sherpa.OfflineRecognizer(config)

    return recognizer


@lru_cache(maxsize=10)
def _get_alimeeting_pre_trained_model(
    repo_id: str,
    decoding_method: str,
    num_active_paths: int,
):
    assert repo_id in [
        "desh2608/icefall-asr-alimeeting-pruned-transducer-stateless7",
        "luomingshuang/icefall_asr_alimeeting_pruned_transducer_stateless2",
    ], repo_id

    if repo_id == "desh2608/icefall-asr-alimeeting-pruned-transducer-stateless7":
        filename = "cpu_jit.pt"
    elif repo_id == "luomingshuang/icefall_asr_alimeeting_pruned_transducer_stateless2":
        filename = "cpu_jit_torch_1.7.1.pt"

    nn_model = _get_nn_model_filename(
        repo_id=repo_id,
        filename=filename,
    )
    tokens = _get_token_filename(repo_id=repo_id)

    feat_config = sherpa.FeatureConfig()
    feat_config.fbank_opts.frame_opts.samp_freq = sample_rate
    feat_config.fbank_opts.mel_opts.num_bins = 80
    feat_config.fbank_opts.frame_opts.dither = 0

    config = sherpa.OfflineRecognizerConfig(
        nn_model=nn_model,
        tokens=tokens,
        use_gpu=False,
        feat_config=feat_config,
        decoding_method=decoding_method,
        num_active_paths=num_active_paths,
    )

    recognizer = sherpa.OfflineRecognizer(config)

    return recognizer


@lru_cache(maxsize=10)
def _get_wenet_model(
    repo_id: str,
    decoding_method: str,
    num_active_paths: int,
):
    assert repo_id in [
        "csukuangfj/wenet-chinese-model",
        "csukuangfj/wenet-english-model",
    ], repo_id

    nn_model = _get_nn_model_filename(
        repo_id=repo_id,
        filename="final.zip",
        subfolder=".",
    )
    tokens = _get_token_filename(
        repo_id=repo_id,
        filename="units.txt",
        subfolder=".",
    )

    feat_config = sherpa.FeatureConfig(normalize_samples=False)
    feat_config.fbank_opts.frame_opts.samp_freq = sample_rate
    feat_config.fbank_opts.mel_opts.num_bins = 80
    feat_config.fbank_opts.frame_opts.dither = 0

    config = sherpa.OfflineRecognizerConfig(
        nn_model=nn_model,
        tokens=tokens,
        use_gpu=False,
        feat_config=feat_config,
        decoding_method=decoding_method,
        num_active_paths=num_active_paths,
    )

    recognizer = sherpa.OfflineRecognizer(config)

    return recognizer


@lru_cache(maxsize=10)
def _get_aidatatang_200zh_pretrained_mode(
    repo_id: str,
    decoding_method: str,
    num_active_paths: int,
):
    assert repo_id in [
        "luomingshuang/icefall_asr_aidatatang-200zh_pruned_transducer_stateless2",
    ], repo_id

    nn_model = _get_nn_model_filename(
        repo_id=repo_id,
        filename="cpu_jit_torch.1.7.1.pt",
    )
    tokens = _get_token_filename(repo_id=repo_id)

    feat_config = sherpa.FeatureConfig()
    feat_config.fbank_opts.frame_opts.samp_freq = sample_rate
    feat_config.fbank_opts.mel_opts.num_bins = 80
    feat_config.fbank_opts.frame_opts.dither = 0

    config = sherpa.OfflineRecognizerConfig(
        nn_model=nn_model,
        tokens=tokens,
        use_gpu=False,
        feat_config=feat_config,
        decoding_method=decoding_method,
        num_active_paths=num_active_paths,
    )

    recognizer = sherpa.OfflineRecognizer(config)

    return recognizer


@lru_cache(maxsize=10)
def _get_tibetan_pre_trained_model(
    repo_id: str,
    decoding_method: str,
    num_active_paths: int,
):
    assert repo_id in [
        "syzym/icefall-asr-xbmu-amdo31-pruned-transducer-stateless7-2022-12-02",
        "syzym/icefall-asr-xbmu-amdo31-pruned-transducer-stateless5-2022-11-29",
    ], repo_id

    filename = "cpu_jit.pt"
    if (
        repo_id
        == "syzym/icefall-asr-xbmu-amdo31-pruned-transducer-stateless5-2022-11-29"
    ):
        filename = "cpu_jit-epoch-28-avg-23-torch-1.10.0.pt"

    nn_model = _get_nn_model_filename(
        repo_id=repo_id,
        filename=filename,
    )

    tokens = _get_token_filename(repo_id=repo_id, subfolder="data/lang_bpe_500")

    feat_config = sherpa.FeatureConfig()
    feat_config.fbank_opts.frame_opts.samp_freq = sample_rate
    feat_config.fbank_opts.mel_opts.num_bins = 80
    feat_config.fbank_opts.frame_opts.dither = 0

    config = sherpa.OfflineRecognizerConfig(
        nn_model=nn_model,
        tokens=tokens,
        use_gpu=False,
        feat_config=feat_config,
        decoding_method=decoding_method,
        num_active_paths=num_active_paths,
    )

    recognizer = sherpa.OfflineRecognizer(config)

    return recognizer


@lru_cache(maxsize=10)
def _get_arabic_pre_trained_model(
    repo_id: str,
    decoding_method: str,
    num_active_paths: int,
):
    assert repo_id in [
        "AmirHussein/icefall-asr-mgb2-conformer_ctc-2022-27-06",
    ], repo_id

    nn_model = _get_nn_model_filename(
        repo_id=repo_id,
        filename="cpu_jit.pt",
    )

    tokens = _get_token_filename(repo_id=repo_id, subfolder="data/lang_bpe_5000")

    feat_config = sherpa.FeatureConfig()
    feat_config.fbank_opts.frame_opts.samp_freq = sample_rate
    feat_config.fbank_opts.mel_opts.num_bins = 80
    feat_config.fbank_opts.frame_opts.dither = 0

    config = sherpa.OfflineRecognizerConfig(
        nn_model=nn_model,
        tokens=tokens,
        use_gpu=False,
        feat_config=feat_config,
        decoding_method=decoding_method,
        num_active_paths=num_active_paths,
    )

    recognizer = sherpa.OfflineRecognizer(config)

    return recognizer


@lru_cache(maxsize=10)
def _get_german_pre_trained_model(
    repo_id: str,
    decoding_method: str,
    num_active_paths: int,
):
    assert repo_id in [
        "csukuangfj/wav2vec2.0-torchaudio",
    ], repo_id

    nn_model = _get_nn_model_filename(
        repo_id=repo_id,
        filename="voxpopuli_asr_base_10k_de.pt",
        subfolder=".",
    )

    tokens = _get_token_filename(
        repo_id=repo_id,
        filename="tokens-de.txt",
        subfolder=".",
    )

    config = sherpa.OfflineRecognizerConfig(
        nn_model=nn_model,
        tokens=tokens,
        use_gpu=False,
        decoding_method=decoding_method,
        num_active_paths=num_active_paths,
    )

    recognizer = sherpa.OfflineRecognizer(config)

    return recognizer


@lru_cache(maxsize=10)
def _get_french_pre_trained_model(
    repo_id: str,
    decoding_method: str,
    num_active_paths: int,
):
    assert repo_id in [
        "shaojieli/sherpa-onnx-streaming-zipformer-fr-2023-04-14",
    ], repo_id

    encoder_model = _get_nn_model_filename(
        repo_id=repo_id,
        filename="encoder-epoch-29-avg-9-with-averaged-model.onnx",
        subfolder=".",
    )

    decoder_model = _get_nn_model_filename(
        repo_id=repo_id,
        filename="decoder-epoch-29-avg-9-with-averaged-model.onnx",
        subfolder=".",
    )

    joiner_model = _get_nn_model_filename(
        repo_id=repo_id,
        filename="joiner-epoch-29-avg-9-with-averaged-model.onnx",
        subfolder=".",
    )

    tokens = _get_token_filename(repo_id=repo_id, subfolder=".")

    recognizer = sherpa_onnx.OnlineRecognizer(
        tokens=tokens,
        encoder=encoder_model,
        decoder=decoder_model,
        joiner=joiner_model,
        num_threads=1,
        sample_rate=16000,
        feature_dim=80,
        decoding_method=decoding_method,
        max_active_paths=num_active_paths,
    )

    return recognizer


@lru_cache(maxsize=10)
def _get_japanese_pre_trained_model(
    repo_id: str,
    decoding_method: str,
    num_active_paths: int,
) -> sherpa.OnlineRecognizer:
    repo_id, kind = repo_id.rsplit("-", maxsplit=1)

    assert repo_id in [
        "TeoWenShen/icefall-asr-csj-pruned-transducer-stateless7-streaming-230208"
    ], repo_id
    assert kind in ("fluent", "disfluent"), kind

    encoder_model = _get_nn_model_filename(
        repo_id=repo_id, filename="encoder_jit_trace.pt", subfolder=f"exp_{kind}"
    )

    decoder_model = _get_nn_model_filename(
        repo_id=repo_id, filename="decoder_jit_trace.pt", subfolder=f"exp_{kind}"
    )

    joiner_model = _get_nn_model_filename(
        repo_id=repo_id, filename="joiner_jit_trace.pt", subfolder=f"exp_{kind}"
    )

    tokens = _get_token_filename(repo_id=repo_id)

    feat_config = sherpa.FeatureConfig()
    feat_config.fbank_opts.frame_opts.samp_freq = sample_rate
    feat_config.fbank_opts.mel_opts.num_bins = 80
    feat_config.fbank_opts.frame_opts.dither = 0

    config = sherpa.OnlineRecognizerConfig(
        nn_model="",
        encoder_model=encoder_model,
        decoder_model=decoder_model,
        joiner_model=joiner_model,
        tokens=tokens,
        use_gpu=False,
        feat_config=feat_config,
        decoding_method=decoding_method,
        num_active_paths=num_active_paths,
        chunk_size=32,
    )

    recognizer = sherpa.OnlineRecognizer(config)

    return recognizer


@lru_cache(maxsize=10)
def _get_paraformer_zh_pre_trained_model(
    repo_id: str,
    decoding_method: str,
    num_active_paths: int,
) -> sherpa_onnx.OfflineRecognizer:
    assert repo_id in [
        "csukuangfj/sherpa-onnx-paraformer-zh-2023-03-28",
    ], repo_id

    nn_model = _get_nn_model_filename(
        repo_id=repo_id,
        filename="model.onnx",
        subfolder=".",
    )

    tokens = _get_token_filename(repo_id=repo_id, subfolder=".")

    recognizer = sherpa_onnx.OfflineRecognizer.from_paraformer(
        paraformer=nn_model,
        tokens=tokens,
        num_threads=2,
        sample_rate=sample_rate,
        feature_dim=80,
        decoding_method="greedy_search",
        debug=False,
    )

    return recognizer


chinese_models = {
    "csukuangfj/sherpa-onnx-paraformer-zh-2023-03-28": _get_paraformer_zh_pre_trained_model,
    "luomingshuang/icefall_asr_wenetspeech_pruned_transducer_stateless2": _get_wenetspeech_pre_trained_model,  # noqa
    "desh2608/icefall-asr-alimeeting-pruned-transducer-stateless7": _get_alimeeting_pre_trained_model,
    "yuekai/icefall-asr-aishell2-pruned-transducer-stateless5-A-2022-07-12": _get_aishell2_pretrained_model,  # noqa
    "yuekai/icefall-asr-aishell2-pruned-transducer-stateless5-B-2022-07-12": _get_aishell2_pretrained_model,  # noqa
    "luomingshuang/icefall_asr_aidatatang-200zh_pruned_transducer_stateless2": _get_aidatatang_200zh_pretrained_mode,  # noqa
    "luomingshuang/icefall_asr_alimeeting_pruned_transducer_stateless2": _get_alimeeting_pre_trained_model,  # noqa
    "csukuangfj/wenet-chinese-model": _get_wenet_model,
    #  "csukuangfj/icefall-asr-wenetspeech-lstm-transducer-stateless-2022-10-14": _get_lstm_transducer_model,
}

english_models = {
    "wgb14/icefall-asr-gigaspeech-pruned-transducer-stateless2": _get_gigaspeech_pre_trained_model,  # noqa
    "yfyeung/icefall-asr-multidataset-pruned_transducer_stateless7-2023-05-04": _get_english_model,  # noqa
    "yfyeung/icefall-asr-finetune-mux-pruned_transducer_stateless7-2023-05-19": _get_english_model,  # noqa
    "WeijiZhuang/icefall-asr-librispeech-pruned-transducer-stateless8-2022-12-02": _get_english_model,  # noqa
    "csukuangfj/icefall-asr-librispeech-pruned-transducer-stateless8-2022-11-14": _get_english_model,  # noqa
    "csukuangfj/icefall-asr-librispeech-pruned-transducer-stateless7-2022-11-11": _get_english_model,  # noqa
    "csukuangfj/icefall-asr-librispeech-pruned-transducer-stateless3-2022-05-13": _get_english_model,  # noqa
    "Zengwei/icefall-asr-librispeech-zipformer-large-2023-05-16": _get_english_model,  # noqa
    "Zengwei/icefall-asr-librispeech-zipformer-2023-05-15": _get_english_model,  # noqa
    "Zengwei/icefall-asr-librispeech-zipformer-small-2023-05-16": _get_english_model,  # noqa
    "videodanchik/icefall-asr-tedlium3-conformer-ctc2": _get_english_model,
    "pkufool/icefall_asr_librispeech_conformer_ctc": _get_english_model,
    "WayneWiser/icefall-asr-librispeech-conformer-ctc2-jit-bpe-500-2022-07-21": _get_english_model,
    "csukuangfj/wenet-english-model": _get_wenet_model,
}

chinese_english_mixed_models = {
    "ptrnull/icefall-asr-conv-emformer-transducer-stateless2-zh": _get_chinese_english_mixed_model,
    "luomingshuang/icefall_asr_tal-csasr_pruned_transducer_stateless5": _get_chinese_english_mixed_model,  # noqa
}

tibetan_models = {
    "syzym/icefall-asr-xbmu-amdo31-pruned-transducer-stateless7-2022-12-02": _get_tibetan_pre_trained_model,  # noqa
    "syzym/icefall-asr-xbmu-amdo31-pruned-transducer-stateless5-2022-11-29": _get_tibetan_pre_trained_model,  # noqa
}

arabic_models = {
    "AmirHussein/icefall-asr-mgb2-conformer_ctc-2022-27-06": _get_arabic_pre_trained_model,  # noqa
}

german_models = {
    "csukuangfj/wav2vec2.0-torchaudio": _get_german_pre_trained_model,
}

french_models = {
    "shaojieli/sherpa-onnx-streaming-zipformer-fr-2023-04-14": _get_french_pre_trained_model,
}

japanese_models = {
    "TeoWenShen/icefall-asr-csj-pruned-transducer-stateless7-streaming-230208-fluent": _get_japanese_pre_trained_model,
    "TeoWenShen/icefall-asr-csj-pruned-transducer-stateless7-streaming-230208-disfluent": _get_japanese_pre_trained_model,
}

all_models = {
    **chinese_models,
    **english_models,
    **chinese_english_mixed_models,
    #  **japanese_models,
    **tibetan_models,
    **arabic_models,
    **german_models,
    **french_models,
}

language_to_models = {
    "Chinese": list(chinese_models.keys()),
    "English": list(english_models.keys()),
    "Chinese+English": list(chinese_english_mixed_models.keys()),
    #  "Japanese": list(japanese_models.keys()),
    "Tibetan": list(tibetan_models.keys()),
    "Arabic": list(arabic_models.keys()),
    "German": list(german_models.keys()),
    "French": list(french_models.keys()),
}
