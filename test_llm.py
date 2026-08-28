from llm import LLM


llm = LLM()

answer = llm.generate_answer(
    question="What is the capital of France?",
    context="France is a country in Europe. The capital of France is Paris."
)

print(answer)