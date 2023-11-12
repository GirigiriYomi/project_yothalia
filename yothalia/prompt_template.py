
prompt = """<<SYS>>
指令：{instruction}
<</SYS>>

{history}
用户：{user}
助手："""

prompt_en = """<s>
<<SYS>>: {instruction}
{history}
<<USER>>: {user}
<<ASSISSTANT>>:
"""
prompt_pure = """<s>
你将要扮演一个{instruction}，结合[]里的对话历史并对<>里的内容进行回复，不要刻意强调自己的身份并尽量简洁
[{history}]
<{user}>
"""


def get_template(lang='en'):
    if lang=='en':
        return prompt_en
    if lang=='zh':
        return prompt
    return prompt_pure




