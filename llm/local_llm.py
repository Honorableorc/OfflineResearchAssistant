from llama_cpp import Llama

MODEL_PATH = r"D:\OfflineResearchAssistant\models\qwen2.5-7b-instruct-q5_k_m-00001-of-00002.gguf"

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    verbose=False
)

def generate_response(prompt: str) -> str:

    response = llm.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful AI Research Assistant. "
                    "Answer clearly, accurately, and concisely."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3,
        max_tokens=2000
    )

    return response["choices"][0]["message"]["content"]