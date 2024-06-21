import os
import requests
import gradio as gr
from MyGPT import MyGPT
from loguru import logger
from urllib.parse import urlparse
from config import MODELS, DEFAULT_MODEL, MODELS_TO_MAX_TOKENS

mygpt = MyGPT()


def user_input_controller(user_input, chat_history):
    if not user_input:
        gr.Warning("输入内容不能为空")
        logger.warning("输入内容不能为空")
        return chat_history
    chat_history.append([user_input, None])
    logger.info(f"\n用户输入：{user_input},\n"f"历史信息：{chat_history}")

    return chat_history


def constract_message(user_input, chat_history, model,
                      max_tokens, temperature, stream):
    if not user_input:
        return chat_history

    logger.info(f"\n用户输入: {user_input}, \n"
                f"历史记录: {chat_history}, \n"
                f"使用模型: {model}, \n"
                f"要生成的最大token数: {max_tokens}\n"
                f"温度: {temperature}\n"
                f"是否流式输出: {stream}")

    messages = user_input
    if len(chat_history) > 1:
        messages = []
        for chat in chat_history:
            if chat[0] is not None:
                messages.append({"role": "user", "content": chat[0]})
            if chat[1] is not None:
                messages.append({"role": "user", "content": chat[1]})
    print(messages)

    generate_res = mygpt.get_response(messages, model, max_tokens,
                                      temperature, stream)

    if stream:
        chat_history[-1][1] = ""
        for char in generate_res:
            char_content = char.choices[0].delta.content
            if char_content is not None:
                chat_history[-1][1] += char_content
                yield chat_history
    else:
        chat_history[-1][1] = generate_res
        logger.info(f"历史记录：{chat_history}")
        yield chat_history


def update_max_tokens(model, origin_tokens):
    new_max_tokens = MODELS_TO_MAX_TOKENS.get(model)
    new_max_tokens = new_max_tokens if new_max_tokens else origin_tokens

    new_set_tokens = origin_tokens if origin_tokens <= new_max_tokens else 1000

    new_max_tokens_compnent = gr.Slider(
        minimum=0,
        maximum=new_max_tokens,
        value=new_set_tokens,
        step=1.0,
        label="Max Tokens",
        interactive=True,
    )

    return new_max_tokens_compnent

def user_input_clean(user_input):
    user_input = ""
    return user_input


def download_image(url, folder_path='generate_img'):

    parsed_url = urlparse(url)
    filename = os.path.join(folder_path, os.path.basename(parsed_url.path))

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    response = requests.get(url)
    with open(filename,'wb') as file:
        file.write(response.content)

    return filename


def fn_generate_image(prompt, model, quality, size, style):
    try:
        response = mygpt.generate_image(prompt, model, quality, size, style)
        logger.debug(f"生成响应:{response}")
        url = response.data[0].url
        revised_prompt = response.data[0].revised_prompt
        file_path=download_image(url)
        logger.debug(f"图片路径：{file_path}")
        return file_path, revised_prompt
    except Exception as error:
        print(str(error))
        raise gr.Error(f"生成图片时出错。 {(str(error))}")

def fn_generate_prompt(origin_prompt, requirement):
    prompt = f'''我正在利用AI图像生成模型生成图片，该模型可接收用户对图像的描述（称之为prompt）来生成图片。我写的原始prompt如下：
    `{origin_prompt}`这个prompt生成的图片我不满意，我想做以下更改：`{requirement}`请你基于我的要求重写一个新的prompt：'''
    response = mygpt.get_response(prompt)

    return response


####################### 布置 Gradio画布##########################

with gr.Blocks() as demo:
    gr.Markdown("# <center> HMQ's GPT </center>")
    with gr.Tab(label="Chat"):
        with gr.Row(equal_height=True):
            with gr.Column(scale=6):
                chatbot = gr.Chatbot(label="GPT")
                user_input_textbox = gr.Textbox(label="Input", value="")
                with gr.Row():
                    submit_btn = gr.Button("Submit")
                    clean_btn = gr.Button("Clear", elem_id="cle")
                    clean_history_btn = gr.Button("Clear History", elem_id="cl_his")

            with gr.Column(scale=1):
                with gr.Tab(label="Parameters"):
                    model_select = gr.Dropdown(
                        label="GPT Model",
                        choices=MODELS,
                        value=DEFAULT_MODEL,
                        multiselect=False,
                        interactive=True,
                    )
                    max_tokens_select = gr.Slider(
                        minimum=0,
                        maximum=4096,
                        value=1000,
                        step=1.0,
                        label="Max Tokens",
                        interactive=True,
                    )
                    temperature_slider = gr.Slider(
                        minimum=0,
                        maximum=2,
                        value=0.7,
                        step=0.01,
                        label="Temperature",
                        interactive=True,
                    )
                    stream_radio = gr.Radio(
                        choices=[True, False],
                        label="Stream Output",
                        value=True,
                        interactive=True,
                    )
    with gr.Tab(label="Picture Generating"):
        with gr.Row():
            with gr.Column():
                with gr.Row(variant="Panel"):
                    images_model = gr.Dropdown(
                        choices=["dall-e-3"],
                        label="Image LLM Model",
                        value="dall-e-3"
                    )
                    images_quality = gr.Dropdown(
                        choices=["standard",
                                 "hd"],
                        label="Image Quality",
                        value="standard"
                    )
                    images_size = gr.Dropdown(
                        choices=["1024x1024",
                                 "1024x1792",
                                 "1792x1024"],
                        label="Image Size",
                        value="1024x1024"
                    )
                    images_style = gr.Dropdown(
                        choices=["vivid",
                                 "natural"],
                        label="Image Style",
                        value="vivid"
                    )
                image_prompt_textbox = gr.Textbox(
                    label="prompt",
                    value="cartoon of 三体里面罗辑"
                )
                image_generate_btn = gr.Button("Generate")

                with gr.Row():
                    require_textbox = gr.Textbox(label="Needs", placeholder="您想如何修改这个图片？")
                    new_image_prompt = gr.Button("Updated Prompt")
                new_image_prompt_textbox = gr.Textbox(label="Fixed Prompt")
                new_image_generate_btn = gr.Button("Generate Again")

            with gr.Column():
                output_image = gr.Image(label="Image Output")
                output_revised_prompt = gr.Textbox(label="Auto-Generate Prompt", placeholder="Image Generate Here after click `Generate`")


#################### Monitoring Action #####################################

        model_select.change(
            fn=update_max_tokens,
            inputs=[model_select, max_tokens_select],
            outputs=max_tokens_select,
        )

        user_input_textbox.submit(
            fn=user_input_controller,
            inputs=[
                user_input_textbox,
                chatbot],
            outputs=[chatbot]
        ).then(
            fn=constract_message,
            inputs=[
                user_input_textbox,
                chatbot,
                model_select,
                max_tokens_select,
                temperature_slider,
                stream_radio],
            outputs=[chatbot]
        ).then(
            fn = user_input_clean,
            inputs=[user_input_textbox],
            outputs=[user_input_textbox]
        )

        submit_btn.click(
            fn=user_input_controller,
            inputs=[user_input_textbox, chatbot],
            outputs=[chatbot],
        ).then(
            fn=constract_message,
            inputs=[
                user_input_textbox,
                chatbot,
                model_select,
                max_tokens_select,
                temperature_slider,
                stream_radio,
            ],
            outputs=[chatbot]
        ).then(
            fn = user_input_clean,
            inputs=[user_input_textbox],
            outputs=[user_input_textbox]
        )

        clean_btn.click(lambda: None, None, user_input_textbox, queue=False)
        clean_history_btn.click(lambda: None, None, chatbot, queue=False)

        #图像生成
        image_prompt_textbox.submit(
            fn=fn_generate_image,
            inputs=[
                image_prompt_textbox,
                images_model,
                images_quality,
                images_size,
                images_style],
            outputs=[
                output_image,
                output_revised_prompt]
        )

        image_generate_btn.click(
            fn=fn_generate_image,
            inputs=[image_prompt_textbox,
                    images_model,
                    images_quality,
                    images_size,
                    images_style],
            outputs=[output_image,
                     output_revised_prompt]
        )

        require_textbox.submit(
            fn=fn_generate_prompt,
            inputs=[output_revised_prompt,
                    require_textbox],
            outputs=[new_image_prompt_textbox])

        new_image_prompt.click(
            fn=fn_generate_image,
            inputs=[output_revised_prompt,
                    require_textbox],
            outputs=[new_image_prompt_textbox])

        new_image_generate_btn.click(
            fn=fn_generate_image,
            inputs=[new_image_prompt_textbox,
                    images_model,
                    images_quality,
                    images_size,
                    images_style],
            outputs=[output_image,
                     output_revised_prompt]
        )



demo.queue().launch()
