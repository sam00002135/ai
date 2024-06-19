import os
from groq import Groq

# 设置API key
api_key = os.environ.get(
    "GROQ_API_KEY") or "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"

client = Groq(api_key=api_key)


def chat_with_model(prompt):
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama3-8b-8192",
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return str(e)


if __name__ == "__main__":
    print("開始聊天吧！（输入 'exit' 退出）")
    while True:
        user_input = input("你: ")
        if user_input.lower() == "exit":
            break
        response = chat_with_model(user_input)
        print(f"AI: {response}")
