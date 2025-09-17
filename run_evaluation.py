from search_qa import run_search, make_client, search_dense
import json
from tqdm.auto import tqdm

client = make_client()

ground_truth = "Data/ground_truth.json"


def load_data(ground_truth):
    with open(ground_truth, "r", encoding="utf-8") as f:
            ground_truth=json.load(f)
    return ground_truth        



def hit_rate(relevance_total):
    cnt = 0

    for line in relevance_total:
        if True in line:
            cnt = cnt + 1

    return cnt / len(relevance_total)   


def mrr(relevance_total):
    total_score = 0.0

    for line in relevance_total:
        for rank in range(len(line)):
            if line[rank] == True:
                total_score = total_score + 1 / (rank + 1)

    return total_score / len(relevance_total)



def evaluate_search(ground_truth, method):
    relevance_total = []

    for id,questions  in tqdm(ground_truth.items()):
        questions_list=json.loads(questions)
        for q in questions_list:
            results=run_search(method, q, client=client,limit =10)
            relevance = [d.id == id for d in results[0:10]]
            relevance_total.append(relevance)
            

    return {
        'hit_rate': hit_rate(relevance_total),
        'mrr': mrr(relevance_total),
    }



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run search evaluation")
    parser.add_argument("--path",  help="path of the ground truth data" , default="Data/ground_truth.json")
    parser.add_argument("--method",  help="dense, sparse, or hybrid")


    args = parser.parse_args()

    data = load_data(args.path)

    results=evaluate_search(data, args.method)
    print(results)



