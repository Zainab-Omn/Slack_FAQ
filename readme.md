# SlackFAQ RAG Builder

📝 Description

SlackFAQ RAG Builder is a lightweight pipeline that ingests Slack export dumps, extracts frequently asked questions (FAQs), and stores them in a vector database to power Retrieval-Augmented Generation (RAG). This allows users to query historical Slack discussions and receive accurate, context-aware answers in real time.


## Ingestion

Take the FAQ file and ingest it into Qdrant collection


```shell
 pipenv run  python .\Ingest_QA.py --file Data/slack_QA.json  --collection slack_dense --mode dense --skip-existing
```

### Arguments
**--file**

  JSON format (per item):

[
   {

    "channel": "some-channel",
    "thread_ts": "1712345678.123456",
    "qas":
     [
      {
        "asked_by": "alice",
        "answered_by": "bob",
        "question": "Q?",
        "answer": "A."
      }
    ]
  },
  ...
]

**--skip-existing**  to prevent ingesting the same documnet on rerun

**--embed-dim**  default is 768

**--model** default is "jinaai/jina-embeddings-v2-base-en"

**--collection** name of the desired collection default is SLACK_FAQ

**--qdrant-url**  http://localhost:6333

**--mode** you can choose from  dense, sparse, or hybrid   with dense as the default one 

   - sparse with bm25 

   - dense with default model and embeding dim that can be changed

   - hyprid with RRF Fusion method



Requirements:

   ```shell
    pipenv --python 3.12 install "qdrant-client[fastembed]>=1.14.2"
   ```  


## Search 
   to search the collection based on the chosen method 

   ```py
  from search_qa import run_search, make_client, search_dense

  client = make_client()
  run_search("hyprid", query, client=client) 
  ```

  or from cmd
  ```shell
  pipenv run  python  search_qa/search.py dense "what is the schedule?" 
  ```  


  ## Evaluation 

  we run the evaluation using hit_rate and mrr for the 3 methods 

| Method  | Hit Rate           | MRR                 |
|---------|--------------------|---------------------|
| dense   | 0.9794044665012407 | 0.8667810862972156  |
| sparse  | 0.860794046650124  | 0.6393688211430155  |
| hybrid  | 0.979907444168735  | 0.828262338020404   |


to run evlaluation 
``` shell
 pipenv run  python  .\run_evaluation.py dense
 ```







