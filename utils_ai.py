import openai

def ask_ai(prompt, api_key):
    if not api_key:
        return "⚠️ Chave de API não configurada. Vá em Settings e insira sua chave OpenAI."
    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Erro ao consultar IA: {e}" 