from api.agents.retrieval_generation import rag_pipeline

from qdrant_client import QdrantClient
from langsmith import Client

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import IDBasedContextPrecision, IDBasedContextRecall, Faithfulness, ResponseRelevancy


ls_client = Client()
qdrant_client = QdrantClient(url="http://localhost:6333")

ragas_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-5.4-mini"))
ragas_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))


def ragas_context_precision_id_based(run, example):
    sample = SingleTurnSample(
        retrieved_context_ids=run.outputs["retrieved_context_ids"],
        reference_context_ids=example.outputs["reference_context_ids"]
    )

    scorer = IDBasedContextPrecision()

    return scorer.single_turn_score(sample)


def ragas_context_recall_id_based(run, example):
    sample = SingleTurnSample(
        retrieved_context_ids=run.outputs["retrieved_context_ids"],
        reference_context_ids=example.outputs["reference_context_ids"]
    )

    scorer = IDBasedContextRecall()

    return scorer.single_turn_score(sample)


def ragas_faithfulness(run, example):

    sample = SingleTurnSample(
            user_input=run.outputs["question"],
            response=run.outputs["answer"],
            retrieved_contexts=run.outputs["retrieved_context"]
        )

    scorer = Faithfulness(llm=ragas_llm)
    
    return scorer.single_turn_score(sample)


def ragas_relevancy(run, example):

    sample = SingleTurnSample(
            user_input=run.outputs["question"],
            response=run.outputs["answer"],
            retrieved_contexts=run.outputs["retrieved_context"]
        )

    scorer = ResponseRelevancy(llm=ragas_llm, embeddings=ragas_embeddings)
    
    return scorer.single_turn_score(sample)


results = ls_client.evaluate(
    lambda x: rag_pipeline(x["question"], qdrant_client),
    data="rag-evaluation-dataset",
    evaluators=[
        ragas_context_precision_id_based,
        ragas_context_recall_id_based,
        ragas_faithfulness,
        ragas_relevancy
    ],
    experiment_prefix="retriever"
)