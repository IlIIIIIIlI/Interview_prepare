#!/usr/bin/env python3
# References:
# https://gradio.app/docs/#dropdown

import base64
import logging
import os
import tempfile
import time
from datetime import datetime

import gradio as gr
import torch
import torchaudio
import urllib.request


from examples import examples
from model import decode, get_pretrained_model, language_to_models, sample_rate

languages = list(language_to_models.keys())

# 将输入的音频文件转换为.wav格式。它使用ffmpeg工具将输入文件转换为16kHz的.wav文件，然后对转换后的文件进行base64编码。
def convert_to_wav(in_filename: str) -> str:
    """Convert the input audio file to a wave file"""
    out_filename = in_filename + ".wav"
    logging.info(f"Converting '{in_filename}' to '{out_filename}'")
    _ = os.system(f"ffmpeg -hide_banner -i '{in_filename}' -ar 16000 '{out_filename}'")
    _ = os.system(
        f"ffmpeg -hide_banner -loglevel error -i '{in_filename}' -ar 16000 '{out_filename}.flac'"
    )

    with open(out_filename + ".flac", "rb") as f:
        s = "\n" + out_filename + "\n"
        s += base64.b64encode(f.read()).decode()
        logging.info(s)

    return out_filename

# 函数build_html_output用于构建HTML格式的输出结果。
def build_html_output(s: str, style: str = "result_item_success"):
    return f"""
    <div class='result'>
        <div class='result_item {style}'>
          {s}
        </div>
    </div>
    """

# 负责处理用户通过麦克风录制的音频文件。它会检查录音是否存在，然后调用process函数进行处理。如果处理过程中出现错误，它会捕获异常并返回错误信息。
def process_microphone(
    language: str,
    repo_id: str,
    decoding_method: str,
    num_active_paths: int,
    in_filename: str,
):
    if in_filename is None or in_filename == "":
        return "", build_html_output(
            "Please first click 'Record from microphone', speak, "
            "click 'Stop recording', and then "
            "click the button 'submit for recognition'",
            "result_item_error",
        )

    logging.info(f"Processing microphone: {in_filename}")
    try:
        return process(
            in_filename=in_filename,
            language=language,
            repo_id=repo_id,
            decoding_method=decoding_method,
            num_active_paths=num_active_paths,
        )
    except Exception as e:
        logging.info(str(e))
        return "", build_html_output(str(e), "result_item_error")

# 它处理音频输入，执行语音识别，并返回识别的文本和一些元数据信息。它首先调用 convert_to_wav 将音频转换为.wav格式，然后获取当前时间并开始计时。之后，它调用 get_pretrained_model 来获取预训练模型，并使用 decode 函数对音频进行识别。最后，它计算音频的总时长，处理时间以及实时因子（RTF，即处理时间除以音频时长）。
@torch.no_grad()
def process(
    language: str,
    repo_id: str,
    decoding_method: str,
    num_active_paths: int,
    in_filename: str,
):
    logging.info(f"language: {language}")
    logging.info(f"repo_id: {repo_id}")
    logging.info(f"decoding_method: {decoding_method}")
    logging.info(f"num_active_paths: {num_active_paths}")
    logging.info(f"in_filename: {in_filename}")

    filename = convert_to_wav(in_filename)

    now = datetime.now()
    date_time = now.strftime("%Y-%m-%d %H:%M:%S.%f")
    logging.info(f"Started at {date_time}")

    start = time.time()

    recognizer = get_pretrained_model(
        repo_id,
        decoding_method=decoding_method,
        num_active_paths=num_active_paths,
    )

    text = decode(recognizer, filename)

    date_time = now.strftime("%Y-%m-%d %H:%M:%S.%f")
    end = time.time()

    metadata = torchaudio.info(filename)
    duration = metadata.num_frames / sample_rate
    rtf = (end - start) / duration

    logging.info(f"Finished at {date_time} s. Elapsed: {end - start: .3f} s")

    info = f"""
    Wave duration  : {duration: .3f} s <br/>
    Processing time: {end - start: .3f} s <br/>
    RTF: {end - start: .3f}/{duration: .3f} = {rtf:.3f} <br/>
    """
    if rtf > 1:
        info += (
            "<br/>We are loading the model for the first run. "
            "Please run again to measure the real RTF.<br/>"
        )

    logging.info(info)
    logging.info(f"\nrepo_id: {repo_id}\nhyp: {text}")

    return text, build_html_output(info)


title = "# For Interview!!! Fight!"


# css style is copied from
# https://huggingface.co/spaces/alphacep/asr/blob/main/app.py#L113
css = """
.result {display:flex;flex-direction:column}
.result_item {padding:15px;margin-bottom:8px;border-radius:15px;width:100%}
.result_item_success {background-color:mediumaquamarine;color:white;align-self:start}
.result_item_error {background-color:#ff7070;color:white;align-self:start}
"""


def update_model_dropdown(language: str):
    if language in language_to_models:
        choices = language_to_models[language]
        return gr.Dropdown.update(choices=choices, value=choices[0])

    raise ValueError(f"Unsupported language: {language}")


import openai
import gradio as gr

# Load your API key from an environment variable or secret management service
openai.api_key = "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"
# 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


####################################################################################################################
##################################       Prompt1 - ask questions     ###############################################
####################################################################################################################
messages = [
  {
    "role": "system",
    "content": "You are now serving as an HR for a technology company and would like to ask me some interview questions. If you find my answers difficult to understand or unclear, please feel free to ask me to clarify the unclear parts. Alternatively, if you are interested, you can also ask me some follow-up questions. Please do not attempt to correct my answers or provide suggestions. The conversation should simulate the real interview."
  },
]

def process_input(user_message):
    global messages

    # Append user message to conversation
    messages.append({"role": "user", "content": user_message})

    # Call OpenAI API
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo-16k", messages=messages)

    # Get the assistant's message from the response
    assistant_message = response['choices'][0]['message']['content']

    # Append assistant message to conversation
    messages.append({"role": "assistant", "content": assistant_message})

    # Create conversation history
    conversation_history = ""
    for message in messages:
        role = message["role"]
        content = message["content"]
        conversation_history += f"{role.title()}: {content}\n"

    return assistant_message, conversation_history

def generate_download_content():
    global messages

    # Create conversation history
    conversation_history = ""
    for message in messages:
        role = message["role"]
        content = message["content"]
        conversation_history += f"{role.title()}: {content}\n"

    return conversation_history

apple = "The quick brown fox jumps over the lazy cat."
textbox_input = gr.inputs.Textbox(default=apple)
tts_interface = gr.load("huggingface/facebook/fastspeech2-en-ljspeech",
                inputs=textbox_input,
                description="TTS using FastSpeech2",
                title="Text to Speech (TTS)",
                examples=[["The quick brown fox jumps over the lazy dog."]])


####################################################################################################################
##################################       Prompt2 - return result     ###############################################
####################################################################################################################
# results
def langchain_query(txt_file, user_message):
    # Open the file
    if txt_file is None:
        preloaded_text = "what do you think of the conversations below?..."
    else:
        preloaded_text = txt_file.read().decode('utf-8')

    results = [  {
    "role": "system",
    "content": "You are now a professional job analysis analyst. Below, I need you to help me provide advice to the job seeker in the following conversation."
  }, {"role": "user", "content": preloaded_text + "below is the conversation between me and HR, How do you think I performed? Based on our previous conversation, please correct my response to make it more logical, structured, professional and colloquial. My reply may contain some filler words and verbal tics(catchphrase); please provide suggestions for improvement in this area as well, with the aim of meeting Australian workplace standards. If my answer is too short(less than 2 minutes), you should give advices on how to expand my answer." + user_message}]
    
    # Call OpenAI API
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo-16k", messages=results)

    # Get the assistant's message from the response
    assistant_message = response['choices'][0]['message']['content']
  
    return assistant_message

###############################################################################
######################### 页面 ###################################
################################################################
# 创建了一个 gr.Blocks 对象，然后在其中添加了一些组件，如 gr.Markdown，gr.Radio，gr.Dropdown，gr.Slider，gr.Tabs，gr.TabItem，gr.Audio，gr.Button，gr.Textbox，gr.HTML 和 gr.Examples。所有这些组件都被配置为以特定方式响应用户输入。
demo = gr.Blocks(css=css)
with demo:
    # 创建一个Markdown文本区域，用于显示标题。
    gr.Markdown(title)
    # 获取所有可用的语言选项。
    # language_choices = list(language_to_models.keys())
    language_choices = ['English']
    print(language_choices)

    # 接下来的一段代码创建了一个Radio按钮组和一个下拉菜单，用于选择语言和相应的模型。
    language_radio = gr.Radio(
        label="Language",
        choices=language_choices,
        value=language_choices[0],
    )
    # print(f"radio: {language_radio}")
    
    model_dropdown = gr.Dropdown(
        choices=language_to_models[language_choices[0]],
        label="Select a model",
        value=language_to_models[language_choices[0]][0],
    )
    print(f"dropdown: {model_dropdown}")

    # 当用户在Radio按钮组中更改选项时，会更新下拉菜单的内容。
    language_radio.change(
        update_model_dropdown,
        inputs=language_radio,
        outputs=model_dropdown,
    )

    # 创建了另一个Radio按钮组和一个滑块，用户可以选择解码方法，并设置活动路径的数量。
    decoding_method_radio = gr.Radio(
        label="Decoding method",
        choices=["greedy_search", "modified_beam_search"],
        value="greedy_search",
    )
    print(f"decoding_method_radio: {decoding_method_radio}")

    num_active_paths_slider = gr.Slider(
        minimum=1,
        value=4,
        step=1,
        label="Number of active paths for modified_beam_search",
    )
    print(f"num_active_paths_slider: {num_active_paths_slider}")

    
    
##############################################################################################
#########################  主体部分   ##################################
########################################################################
    # 创建一个标签页容器。
    
    with gr.Tabs() as tabs:
        
        with gr.TabItem("Chat with AI"):
            textbox_input = gr.inputs.Textbox(lines=5, placeholder="Type your message here...")
            textbox_output = gr.outputs.Textbox(label="Assistant's response")
            conversation_output = gr.outputs.Textbox(label="Conversation history")
            submit_button = gr.Button("Submit")
            # download_link = gr.outputs.Download(fn=generate_download_content, label="Download conversation")

            submit_button.click(
                process_input,
                inputs=[textbox_input],
                outputs=[textbox_output, conversation_output],
            )
        
        # 用户可以通过麦克风录制音频进行语音识别。
        with gr.TabItem("Record from microphone"):
            gr.TabbedInterface([tts_interface], ["FastSpeech2"])
            microphone = gr.Audio(
                source="microphone",  # Choose between "microphone", "upload"
                type="filepath",
                optional=False,
                label="Record from microphone",
            )

            record_button = gr.Button("Submit for recognition")
            recorded_output = gr.Textbox(label="Recognized speech from recordings")
            recorded_html_info = gr.HTML(label="Info")
            
        # 当用户点击录音按钮或URL按钮时，也会调用相应的处理函数。
        record_button.click(
            process_microphone,
            inputs=[
                language_radio,
                model_dropdown,
                decoding_method_radio,
                num_active_paths_slider,
                microphone,
            ],
            outputs=[recorded_output, recorded_html_info],
        )

        with gr.TabItem("Results"):
            filename_input = gr.inputs.File(label="Upload .txt file", optional=True)
            textbox_input = gr.inputs.Textbox(lines=5, placeholder="put all history here")
            textbox_output = gr.outputs.Textbox(label="Assistant's response")
            submit_button = gr.Button("Submit")

            submit_button.click(
                langchain_query,
                inputs=[filename_input, textbox_input],
                outputs=[textbox_output],
            )

        
    

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)
torch._C._set_graph_executor_optimize(False)

if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    # 设置日志的格式和级别。
    logging.basicConfig(format=formatter, level=logging.INFO)

    # 启动Gradio界面。
    demo.launch()
