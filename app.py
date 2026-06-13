import os
import torch
import librosa
import warnings
import numpy as np
import pandas as pd
import gradio as gr
from model import EvalNet, t_EvalNet
from utils import (
    get_modelist,
    find_files,
    embed,
    ZH2EN,
    MODEL_DIR,
    SAMPLE_RATE,
    HOP_LENGTH,
    TIME_LENGTH,
    TRANSLATE,
    CLASSES,
)

I18N = gr.I18n(
    zh={key: key for key in ZH2EN},
    en=ZH2EN,
)


def logMel(y, sr=SAMPLE_RATE):
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        hop_length=HOP_LENGTH,
        fmin=27.5,
    )
    return librosa.power_to_db(mel, ref=np.max)


def logCqt(y, sr=SAMPLE_RATE):
    cqt = librosa.cqt(
        y,
        sr=sr,
        hop_length=HOP_LENGTH,
        fmin=27.5,
        n_bins=88,
        bins_per_octave=12,
    )
    return ((1.0 / 80.0) * librosa.core.amplitude_to_db(np.abs(cqt), ref=np.max)) + 1.0


def logChroma(y, sr=SAMPLE_RATE):
    chroma = librosa.feature.chroma_stft(
        y=y,
        sr=sr,
        hop_length=HOP_LENGTH,
    )
    return (
        (1.0 / 80.0) * librosa.core.amplitude_to_db(np.abs(chroma), ref=np.max)
    ) + 1.0


def RoW_norm(data):
    common_sum = 0
    square_sum = 0
    tfle = 0
    for i in range(len(data)):
        tfle += (data[i].sum(-1).sum(0) != 0).astype("float").sum()
        common_sum += data[i].sum(-1).sum(-1)
        square_sum += (data[i] ** 2).sum(-1).sum(-1)

    common_avg = common_sum / tfle
    square_avg = square_sum / tfle
    std = np.sqrt(square_avg - common_avg**2)
    return common_avg, std


def norm(data):
    size = data.shape
    avg, std = RoW_norm(data)
    avg = np.tile(avg.reshape((1, -1, 1, 1)), (size[0], 1, size[2], size[3]))
    std = np.tile(std.reshape((1, -1, 1, 1)), (size[0], 1, size[2], size[3]))
    return (data - avg) / std


def chunk_data(f):
    x = []
    xdata = np.transpose(f)
    s = SAMPLE_RATE * TIME_LENGTH // HOP_LENGTH
    length = int(np.ceil((int(len(xdata) / s) + 1) * s))
    app = np.zeros((length - xdata.shape[0], xdata.shape[1]))
    xdata = np.concatenate((xdata, app), 0)
    for i in range(int(length / s)):
        data = xdata[int(i * s) : int(i * s + s)]
        x.append(np.transpose(data[:s, :]))

    return np.array(x)


def load(audio_path: str, converto="mel"):
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    spec = eval("log%s(y, sr)" % converto.capitalize())
    x_spec = chunk_data(spec)
    Xtr_spec = np.expand_dims(x_spec, axis=3)
    return list(norm(Xtr_spec))


def format_second(seconds):
    integer_part = int(seconds)
    decimal_part = round(seconds - integer_part, 3)
    hours, remainder = divmod(integer_part, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}.{decimal_part:.3f}"


def infer(audio_path: str, log_name: str):
    status = "Success"
    filename = result = None
    try:
        if not audio_path:
            raise ValueError("请输入音频!")

        backbone = "_".join(log_name.split("_")[:-1])
        spec = log_name.split("_")[-1]
        input = load(audio_path, converto=spec)
        dur = librosa.get_duration(path=audio_path)
        frames_per_3s = input[0].shape[1]
        if "vit" in backbone or "swin" in backbone:
            eval_net = t_EvalNet(
                backbone,
                len(TRANSLATE),
                frames_per_3s,
                weight_path=f"{MODEL_DIR}/{log_name}.pt",
            )

        else:
            eval_net = EvalNet(
                backbone,
                len(TRANSLATE),
                frames_per_3s,
                weight_path=f"{MODEL_DIR}/{log_name}.pt",
            )

        input_size = eval_net.get_input_size()
        embeded_input = embed(input, input_size)
        output = []
        for x in embeded_input:
            output.append(eval_net.forward(x))

        index = 0
        outputs = []
        for y in output:
            preds = list(y.T)
            for pred in preds:
                start = index * TIME_LENGTH / frames_per_3s
                if start > dur:
                    break

                to = (index + 1) * TIME_LENGTH / frames_per_3s
                outputs.append(
                    {
                        I18N("帧数"): f"{format_second(start)} - {format_second(to)}",
                        I18N("技法"): I18N(
                            TRANSLATE[CLASSES[torch.argmax(pred).item()]]
                        ),
                    }
                )
                index += 1

        filename = os.path.basename(audio_path)
        result = pd.DataFrame(outputs)

    except Exception as e:
        status = f"{e}"

    return status, filename, result


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    models = get_modelist(assign_model="VGG19_mel")
    examples = []
    example_wavs = find_files()
    for wav in example_wavs:
        examples.append([wav, models[0]])

    with gr.Blocks() as demo:
        gr.Interface(
            fn=infer,
            inputs=[
                gr.Audio(label=I18N("上传录音"), type="filepath"),
                gr.Dropdown(choices=models, label=I18N("选择模型"), value=models[0]),
            ],
            outputs=[
                gr.Textbox(label=I18N("状态栏"), buttons=["copy"]),
                gr.Textbox(label=I18N("音频文件名"), buttons=["copy"]),
                gr.Dataframe(label=I18N("古筝演奏技法逐帧检测")),
            ],
            examples=examples,
            cache_examples=False,
            flagging_mode="never",
            title=I18N("建议录音时长不要过长"),
        )

        gr.Markdown(f"# {I18N('引用')}" + """
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
            ```""")

    demo.launch(
        theme=gr.themes.Ocean(),
        css="#gradio-share-link-button-0, thead { display: none; }",
        ssr_mode=False,
        i18n=I18N,
    )
