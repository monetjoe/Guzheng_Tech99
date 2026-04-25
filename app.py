import os
import torch
import shutil
import librosa
import warnings
import numpy as np
import gradio as gr
import librosa.display
import matplotlib.pyplot as plt
from model import EvalNet
from utils import (
    get_modelist,
    find_wav_files,
    embed_img,
    _L,
    EN_US,
    SAMPLE_RATE,
    TEMP_DIR,
    TRANSLATE,
    CLASSES,
)


def circular_padding(y: np.ndarray, sr: int, dur=3):
    if len(y) >= sr * dur:
        return y[: sr * dur]

    size = sr * dur // len(y) + int((sr * dur) % len(y) > 0)
    arrays = []
    for _ in range(size):
        arrays.append(y)

    y = np.hstack(arrays)
    return y[: sr * dur]


def wav2mel(audio_path: str):
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    y = circular_padding(y, sr)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    librosa.display.specshow(log_mel_spec)
    plt.axis("off")
    plt.savefig(
        f"{TEMP_DIR}/output.jpg",
        bbox_inches="tight",
        pad_inches=0.0,
    )
    plt.close()


def wav2cqt(audio_path: str):
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    y = circular_padding(y, sr)
    cqt_spec = librosa.cqt(y=y, sr=sr)
    log_cqt_spec = librosa.power_to_db(np.abs(cqt_spec) ** 2, ref=np.max)
    librosa.display.specshow(log_cqt_spec)
    plt.axis("off")
    plt.savefig(
        f"{TEMP_DIR}/output.jpg",
        bbox_inches="tight",
        pad_inches=0.0,
    )
    plt.close()


def wav2chroma(audio_path: str):
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    y = circular_padding(y, sr)
    chroma_spec = librosa.feature.chroma_stft(y=y, sr=sr)
    log_chroma_spec = librosa.power_to_db(np.abs(chroma_spec) ** 2, ref=np.max)
    librosa.display.specshow(log_chroma_spec)
    plt.axis("off")
    plt.savefig(
        f"{TEMP_DIR}/output.jpg",
        bbox_inches="tight",
        pad_inches=0.0,
    )
    plt.close()


def infer(wav_path: str, log_name: str, folder_path=TEMP_DIR):
    status = "Success"
    filename = result = None
    try:
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)

        if not wav_path:
            return None, "请输入音频!"

        spec = log_name.split("_")[-3]
        os.makedirs(folder_path, exist_ok=True)
        model = EvalNet(log_name, len(TRANSLATE)).model
        eval("wav2%s" % spec)(wav_path)
        input = embed_img(f"{folder_path}/output.jpg")
        output: torch.Tensor = model(input)
        pred_id = torch.max(output.data, 1)[1]
        filename = os.path.basename(wav_path)
        result = (
            CLASSES[pred_id].capitalize()
            if EN_US
            else f"{TRANSLATE[CLASSES[pred_id]]} ({CLASSES[pred_id].capitalize()})"
        )

    except Exception as e:
        status = f"{e}"

    return status, filename, result


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    models = get_modelist(assign_model="swin_t_mel")
    examples = []
    example_wavs = find_wav_files()
    for wav in example_wavs:
        examples.append([wav, models[0]])

    with gr.Blocks() as demo:
        gr.Interface(
            fn=infer,
            inputs=[
                gr.Audio(label=_L("上传录音"), type="filepath"),
                gr.Dropdown(choices=models, label=_L("选择模型"), value=models[0]),
            ],
            outputs=[
                gr.Textbox(label=_L("状态栏"), buttons=["copy"]),
                gr.Textbox(label=_L("音频文件名"), buttons=["copy"]),
                gr.Textbox(label=_L("演奏技法识别"), buttons=["copy"]),
            ],
            examples=examples,
            cache_examples=False,
            flagging_mode="never",
            title=_L("建议录音时长保持在 3s 左右"),
        )

        gr.Markdown(
            f"# {_L('引用')}"
            + """
            ```bibtex
            @article{Zhou-2025,
                author  = {Monan Zhou and Shenyang Xu and Zhaorui Liu and Zhaowen Wang and Feng Yu and Wei Li and Baoqiang Han},
                title   = {CCMusic: An Open and Diverse Database for Chinese Music Information Retrieval Research},
                journal = {Transactions of the International Society for Music Information Retrieval},
                volume  = {8},
                number  = {1},
                pages   = {22--38},
                month   = {Mar},
                year    = {2025},
                url     = {https://doi.org/10.5334/tismir.194},
                doi     = {10.5334/tismir.194}
            }
            ```"""
        )

    demo.launch(css="#gradio-share-link-button-0 { display: none; }")
