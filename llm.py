import ollama


class LLM:

    def __init__(self, model="qwen2.5:3b"):
        self.model = model

    def generate_answer(self, question, context):

        prompt = f"""
You are a helpful assistant.

Answer the question using only the provided context.

If the answer is not available in the context, say:
"I could not find the answer in the provided documents."

Context:
{context}

Question:
{question}

Answer:
"""

        response = ollama.generate(
            model=self.model,
            prompt=prompt
        )

        return response["response"]