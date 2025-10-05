from search_qa import run_search, make_client, search_dense
import json
from tqdm.auto import tqdm


from dotenv import load_dotenv
from dotenv import dotenv_values

from openai import OpenAI


db_client = make_client()

config = dotenv_values(".env")
api_key = config.get("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

def build_prompt(query, search_results):
    prompt_template = """
You're a course teaching assistant. Answer the QUESTION based on the CONTEXT from the FAQ database.
Use only the facts from the CONTEXT when answering the QUESTION.

    QUESTION: {question}

    CONTEXT: 
    {context}
    """.strip()

    context = ""
    
    for doc in search_results:
        context = context + f"question: {doc.payload['question']}\n answer: {doc.payload['answer']}\n\n"
    
    prompt = prompt_template.format(question=query, context=context).strip()
    return prompt



def llm(prompt, model='gpt-4o-mini'):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    
    answer = response.choices[0].message.content

    # token counts (safe defaults if field missing)
    tokens_in = response.usage.prompt_tokens if hasattr(response, "usage") else 0
    tokens_out = response.usage.completion_tokens if hasattr(response, "usage") else 0

    return {
        "answer": answer,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
    }


def rag(query,method='dense', model='gpt-4o-mini',limit=5) -> str:
    search_results = run_search(method, query, client=db_client,limit=limit)
    prompt = build_prompt(query, search_results)
    answer = llm(prompt, model=model)
    return {
        "answer": answer["answer"],
        "tokens_in": answer["tokens_in"],
        "tokens_out": answer["tokens_out"],
    }


def calculate_llm_cost(tokens_in: int, tokens_out: int) -> float:
    """
    Calculate the USD cost of a gpt-4o-mini call.

    Args:
        tokens_in (int): Number of input tokens.
        tokens_out (int): Number of output tokens.

    Returns:
        float: Cost in USD, rounded to 6 decimal places.
    """
    # pricing per 1K tokens
    INPUT_PRICE = 0.000150
    OUTPUT_PRICE = 0.000600

    cost = (tokens_in / 1000) * INPUT_PRICE + (tokens_out / 1000) * OUTPUT_PRICE
    return round(cost, 6)



def compute_relevancy(question, answer):
    prompt_template = """
    You are an expert evaluator for a Retrieval-Augmented Generation (RAG) system.
    Your task is to analyze the relevance of the generated answer to the given question.
    Based on the relevance of the generated answer, you will classify it
    as "NON_RELEVANT", "PARTLY_RELEVANT", or "RELEVANT".

    Here is the data for evaluation:

    Question: {question}
    Generated Answer: {answer}

    Please analyze the content and context of the generated answer in relation to the question
    and provide your evaluation in parsable JSON without using code blocks:

    {{
      "Relevance": "NON_RELEVANT" | "PARTLY_RELEVANT" | "RELEVANT",
      "Explanation": "[Provide a brief explanation for your evaluation]"
    }}
    """.strip()

    prompt = prompt_template.format(question=question, answer=answer)
    result=llm(prompt)
    
    try:
        json_eval = json.loads(result["answer"])
    
        return {
        "Relevance":  json_eval['Relevance'],
        "Explanation": json_eval['Explanation'],
         }
    except json.JSONDecodeError:
        return {
        "Relevance":  "UNKNOWN",
        "Explanation": "Failed to parse evaluation",
         }



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Rag")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--method" , default='dense', help="Search method")
    parser.add_argument("--channel", default="#course-llm-zoomcamp", help="Slack channel")
    parser.add_argument("--limit", type=int, default=1, help="Number of results")


    args = parser.parse_args()

    print(rag(args.query,method=args.method,limit=args.limit))
 