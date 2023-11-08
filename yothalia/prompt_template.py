
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

def get_template(lang='en'):
    if lang=='en':
        return prompt_en
    return prompt




