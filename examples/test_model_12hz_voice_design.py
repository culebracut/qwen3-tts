# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import time
import torch
import soundfile as sf

from qwen_tts import Qwen3TTSModel


def main():
    device = "cuda:0"
    MODEL_PATH = "/data/models/qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign/"

    tts = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        device_map=device,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    # -------- Single --------
    # torch.cuda.synchronize()
    # t0 = time.time()

    # wavs, sr = tts.generate_voice_design(
    #     text="哥哥，你回来啦，人家等了你好久好久了，要抱抱！",
    #     language="Chinese",
    #     instruct="体现撒娇稚嫩的萝莉女声，音调偏高且起伏明显，营造出黏人、做作又刻意卖萌的听觉效果。",
    # )

    # torch.cuda.synchronize()
    # t1 = time.time()
    # print(f"[VoiceDesign Single] time: {t1 - t0:.3f}s")

    # sf.write("/data/audio/output/qwen/design/qwen3_tts_test_voice_design_single.wav", wavs[0], sr)

    # -------- Batch --------
    texts = [
        "哥哥，你回来啦，人家等了你好久好久了，要抱抱！",
        #"It's in the top drawer... wait, it's empty? No way, that's impossible! I'm sure I put it there!"
        "Four score and seven years ago our fathers brought forth on this continent a new nation, conceived in liberty, and dedicated to the proposition that all men are created equal. “Now we are engaged in a great civil war, testing whether that nation, or any nation so conceived and so dedicated, can long endure. We are met on a great battlefield of that war.",
        "Four score and seven years ago our fathers brought forth on this continent a new nation, conceived in liberty, and dedicated to the proposition that all men are created equal. “Now we are engaged in a great civil war, testing whether that nation, or any nation so conceived and so dedicated, can long endure. We are met on a great battlefield of that war.",
        "Four score and seven years ago our fathers brought forth on this continent a new nation, conceived in liberty, and dedicated to the proposition that all men are created equal. “Now we are engaged in a great civil war, testing whether that nation, or any nation so conceived and so dedicated, can long endure. We are met on a great battlefield of that war.",
        #"Four score and seven years ago our fathers brought forth on this continent a new nation, conceived in liberty, and dedicated to the proposition that all men are created equal. “Now we are engaged in a great civil war, testing whether that nation, or any nation so conceived and so dedicated, can long endure. We are met on a great battlefield of that war."
    ]
    languages = ["Chinese", "English","English","English"]
    instructs = [
        "体现撒娇稚嫩的萝莉女声，音调偏高且起伏明显，营造出黏人、做作又刻意卖萌的听觉效果。",
        "A male voice with a rhythmic, staccato delivery and sharp tonal shifts. The pitch is high and unstable, sounding incredulous and increasingly frantic with audible breathiness.",
        "Speak in an happy tone, but with a hint of panic beginning to creep into your voice.",
        "A mid-pitched male voice with a storytelling flair, using dramatic pauses and varied intonation."
    ]

    torch.cuda.synchronize()
    t0 = time.time()

    wavs, sr = tts.generate_voice_design(
        text=texts,
        language=languages,
        instruct=instructs,
        max_new_tokens=2048,
    )

    torch.cuda.synchronize()
    t1 = time.time()
    print(f"[VoiceDesign Batch] time: {t1 - t0:.3f}s")

    for i, w in enumerate(wavs):
        sf.write(f"/data/audio/output/qwen/design/qwen3_tts_test_voice_design_batch_{i}.wav", w, sr)


if __name__ == "__main__":
    main()