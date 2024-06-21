from openai import OpenAI
from loguru import logger
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv('api.env'))


class MyGPT:
    def __init__(self):
        self.client = OpenAI()

    def get_response(self, messages, model="gpt-3.5-turbo", max_tokens=1000, temperature=0.7, stream=False):
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        elif not isinstance(messages, list):
            return "无效的 ‘messages’ 类型，它应该是一个字符串或者消息队列"

        completion = self.client.chat.completions.create(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            stream=stream,
            temperature=temperature
        )

        if stream:
            return completion

        logger.debug(completion.choices[0].message.content)
        logger.info(f"总token数：{completion.usage.total_tokens}")
        return completion.choices[0].message.content

    def get_embedding(self, input):
        """
        Embedding API：https://platform.openai.com/docs/api-reference/embeddings

        Creates an embedding vector representing the input text.
        创建表示输入文本的嵌入向量。

        Args:
            input: 输入要嵌入的文本，编码为字符串或标记数组。若要在单个请求中嵌入多个输入，请传递字符串数组或token数组数组。
                输入不得超过模型的最大输入标记数（8192 text-embedding-ada-002 个标记），不能为空字符串，任何数组的维数必须小于或等于 2048。

        Returns:
            嵌入对象的列表。
        """
        response = self.client.embeddings.create(
            input=input,
            model='text-embedding-ada-002',
        )
        embeddings = [data.embedding for data in response.data]
        return embeddings

    def generate_image(self, prompt, model="dall-e-3", quality="standard", size="1024x1024", style="vivid", n=1,
                       response_format="url"):
        """
        Images API: https://platform.openai.com/docs/api-reference/images

        Creates an image given a prompt.
        在给定提示的情况下创建图像。

        Args:
            prompt: 所需图像的文本描述。dall-e-2 的最大长度为 1000 个字符，dall-e-3 的最大长度为 4000 个字符。
            model: 用于图像生成的模型。
            quality: 将生成的图像的质量。hd 创建具有更精细细节和更高一致性的图像。此参数仅支持 dall-e-3。
            size: 生成图像的大小。
                对于 dall-e-2，必须是 256x256、512x512 或 1024x1024 之一。
                对于 dall-e-3 模型，必须是 1024x1024、1792x1024 或 1024x1792 之一。
            style: 生成图像的风格。必须是生动或自然。生动 "使模型偏向于生成超真实和戏剧化的图像。自然 "会使模型生成更自然、不那么超真实的图像。此参数仅支持 dall-e-3。
            n: 要生成的图像数。必须介于 1 和 10 之间。对于 dall-e-3，仅 n=1 受支持。
            response_format: 返回生成的图像的格式。必须是 url 或 b64_json 之一。URL 仅在图像生成后的 60 分钟内有效。

        Returns:
            返回图像对象的列表。
        """
        response = self.client.images.generate(
            prompt=prompt,
            model=model,
            quality=quality,
            size=size,
            style=style,
            n=n,
            response_format=response_format,
        )
        logger.info(f"优化后的prompt: {response.data[0].revised_prompt}")
        return response



if __name__ == "__main__":
    # 测试
    mygpt = MyGPT()

    # prompt
    # prompt = '你好'
    # response = mygpt.get_response(prompt, temperature=1)
    # print(response)

    #
    # # messages
    # messages = [
    #     {'role': 'user', 'content': '什么是大模型'},
    # ]
    # response = mygpt.get_completion(messages, temperature=1)
    # print(response)

    # 嵌入
    # vectors = mygpt.get_embedding("input text")
    # print(len(vectors), len(vectors[0]))
    # # 1 1536
    #
    # vectors = mygpt.get_embedding(["input text 1", "input text 2"])
    # print(len(vectors), len(vectors[0]))
    # # 2 1536

    # # 图像生成
    prompt = 'cartoon of 蓝天白云，动物园'
    response = mygpt.generate_image(prompt)
    print(response)
