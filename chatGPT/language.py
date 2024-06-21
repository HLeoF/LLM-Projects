def set_language(language):
    if language == "English":
        return {
            "tab1": "Parameters",
            "tab2": "Settings",
            "sub_bt": "Submit",
            "cle_bt": "Clear",
            "model_lab": "GPT Model",
            "max_tokens_lab": "Max Tokens",
            "temp_lab": "Temperature",
            "stream_lab": "Stream Output",
            "language_lab": "Select Language",
        }
    elif language == "中文":
        return{
            "tab1": "参数",
            "tab2": "设置",
            "sub_bt": "提交",
            "cle_bt": "清除",
            "model_lab": "GPT 模型",
            "max_tokens_lab": "最大 Tokens",
            "temp_lab": "温度",
            "stream_lab": "流式输出",
            "language_lab": "选择语言",
        }