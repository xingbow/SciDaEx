import os
import logging
from typing import List, Dict, Optional
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRelevancyMetric, AnswerRelevancyMetric, FaithfulnessMetric
from deepeval import evaluate

try:
    import globalVariable as GV
except:
    from app.dataService import globalVariable as GV

os.environ["OPENAI_API_KEY"] = GV.openai_key
def llm_evaluate_deepeval(metrics=['faithfulness', 'answer_relevancy', 'contextual_relevancy'], question=None, answer=None, contexts=None):
    """
    Evaluates the LLM response using specified metrics.

    Args:
        metrics (List[str]): List of metrics to evaluate.
        question (Optional[str]): The input question.
        answer (Optional[str]): The LLM's answer.
        contexts (Optional[str]): The context provided for the answer.

    Returns:
        Dict[str, float]: A dictionary of metric names and their scores.

    Raises:
        ValueError: If required inputs are missing or invalid metrics are provided.
    """
    evaluate_case = LLMTestCase(
        input=question,
        actual_output=answer,
        retrieval_context=[contexts]
    )

    evaluate_metrics = []
    for metric in metrics:
        if metric == "faithfulness":
            evaluate_metrics.append(FaithfulnessMetric())
        if metric == "answer_relevancy":
            evaluate_metrics.append(AnswerRelevancyMetric())
        if metric == "contextual_relevancy":
            evaluate_metrics.append(ContextualRelevancyMetric())

    evaluate_response = evaluate(
        test_cases=[evaluate_case],
        metrics=evaluate_metrics,
        print_results=False
    )

    evaluate_result = {}
    for individual_result in evaluate_response:
        for i in range(0, len(individual_result.metrics_metadata)):
            evaluate_result[metrics[i]] = individual_result.metrics_metadata[i].score
    
    return evaluate_result

if __name__ == "__main__":
   # Test cases
    test_cases = [
        {
            "question": "What is multi-modality in AI?",
            "contexts": "Multi-modality in AI refers to the integration and analysis of information from multiple different modes or modalities, such as text, images, audio, video, or sensor data.",
            "answer": "Multi-modality in AI is the integration and analysis of information from multiple different modes or types of data, like text, images, audio, and video."
        },
        # Add more test cases here
    ]
    for case in test_cases:
        result = llm_evaluate_deepeval(
                    question=case["question"],
                    answer=case["answer"],
                    contexts=case["contexts"]
                )
        print(f"evaluation result: {result}")