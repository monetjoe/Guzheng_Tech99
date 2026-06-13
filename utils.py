import os
import torch
import numpy as np
from torchvision.transforms import Compose, Resize, Normalize

EN_US = os.getenv("LANG") != "zh_CN.UTF-8"

ZH2EN = {
    "上传录音": "Upload a recording",
    "选择模型": "Select a model",
    "状态栏": "Status",
    "音频文件名": "Audio filename",
    "古筝演奏技法逐帧检测": "Frame-level guzheng playing technique detection",
    "建议录音时长不要过长": "It is suggested that the recording time should not be too long",
    "引用": "Cite",
    "颤音": "Vibrato",
    "拨弦": "Plucks",
    "上滑音": "Upward Portamento",
    "下滑音": "Downward Portamento",
    "花指\刮奏\连抹\连托": "Glissando",
    "摇指": "Tremolo",
    "点音": "Point Note",
    "帧数": "Frame",
    "技法": "Tech",
}

if EN_US:
    import huggingface_hub

    MODEL_DIR = huggingface_hub.snapshot_download(
        "ccmusic-database/Guzheng_Tech99",
        cache_dir="./__pycache__",
    )

else:
    import modelscope

    MODEL_DIR = modelscope.snapshot_download(
        "ccmusic-database/Guzheng_Tech99",
        cache_dir="./__pycache__",
    )


def _L(zh_txt: str):
    return ZH2EN[zh_txt] if EN_US else zh_txt


TRANSLATE = {
    "chanyin": _L("颤音"),  # Vibrato
    "boxian": _L("拨弦"),  # Plucks
    "shanghua": _L("上滑音"),  # Upward Portamento
    "xiahua": _L("下滑音"),  # Downward Portamento
    "huazhi/guazou/lianmo/liantuo": _L("花指\刮奏\连抹\连托"),  # Glissando
    "yaozhi": _L("摇指"),  # Tremolo
    "dianyin": _L("点音"),  # Point Note
}
CLASSES = list(TRANSLATE.keys())
TEMP_DIR = "./__pycache__/tmp"
SAMPLE_RATE = 44100
HOP_LENGTH = 512
TIME_LENGTH = 3


def toCUDA(x):
    if hasattr(x, "cuda"):
        if torch.cuda.is_available():
            return x.cuda()

    return x


def find_files(folder_path=f"{MODEL_DIR}/examples", ext=".flac"):
    audio_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(ext):
                file_path = os.path.join(root, file)
                audio_files.append(file_path)

    return audio_files


def get_modelist(model_dir=MODEL_DIR, assign_model=""):
    pt_files = []
    for _, _, files in os.walk(model_dir):
        for file in files:
            if file.endswith(".pt"):
                model = os.path.basename(file)[:-3]
                if assign_model and assign_model.lower() in model:
                    pt_files.insert(0, model)
                else:
                    pt_files.append(model)

    return pt_files


def embed(input: list, img_size: int):
    compose = Compose(
        [
            Resize([img_size, img_size]),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    inputs = []
    for x in input:
        x = np.array(x).transpose(2, 0, 1)
        x = torch.from_numpy(x).repeat(3, 1, 1)
        x = torch.tensor(np.array([compose(x).float()]))
        inputs.append(toCUDA(x))

    return inputs
