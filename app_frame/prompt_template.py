
prompt = """[INST] <<SYS>>
指令：{instruction}
<</SYS>>

{history}
用户：{user}
助手："""

def get_template():
    return prompt



