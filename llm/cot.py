# llm/cot.py

from typing import Dict, Optional, Tuple
from .llm import LLM
import json
from datetime import datetime

class ChainOfThought:
    def __init__(
        self,
        generator_llm: LLM,
        evaluator_llm: Optional[LLM] = None,
        threshold: float = 95.0,
        max_iterations: int = 5,
        log_file: str = "cot_logs.json"
    ):
        """
        Initialize Chain of Thought processor.
        
        Parameters:
        - generator_llm (LLM): LLM instance for generating responses
        - evaluator_llm (LLM, optional): LLM instance for evaluating responses
                                       If None, uses generator_llm
        - threshold (float): Score threshold to stop iterations (0-100)
        - max_iterations (int): Maximum number of improvement iterations
        - log_file (str): File to store execution logs
        """
        self.generator = generator_llm
        self.evaluator = evaluator_llm or generator_llm
        self.threshold = threshold
        self.max_iterations = max_iterations
        self.log_file = log_file
        
    def _create_evaluation_prompt(self, question: str, response: str) -> str:
        """Create prompt for the evaluator."""
        return f"""Please evaluate the following response to the given question.
        Provide a score from 1 to 100 and detailed feedback for improvement.
        
        Question: {question}
        
        Response: {response}
        
        Evaluate the response based on:
        1. Accuracy and correctness
        2. Completeness
        3. Clarity and coherence
        4. Relevance to the question
        
        Format your response as JSON with the following structure:
        {{
            "score": <number between 1 and 100>,
            "feedback": "<detailed feedback>",
            "strengths": ["<strength1>", "<strength2>", ...],
            "areas_for_improvement": ["<area1>", "<area2>", ...]
        }}
        """

    def _create_improvement_prompt(
        self, 
        question: str, 
        previous_response: str, 
        evaluation: Dict
    ) -> str:
        """Create prompt for improving the response."""
        return f"""Please improve the following response based on the evaluation provided.

        Original Question: {question}
        
        Previous Response: {previous_response}
        
        Evaluation Score: {evaluation.get('score')}
        
        Feedback: {evaluation.get('feedback')}
        
        Strengths:
        {json.dumps(evaluation.get('strengths', []), indent=2)}
        
        Areas for Improvement:
        {json.dumps(evaluation.get('areas_for_improvement', []), indent=2)}
        
        Please provide an improved response addressing the feedback and areas for improvement
        while maintaining the identified strengths.
        """

    def _evaluate_response(self, question: str, response: str) -> Dict:
        """Get evaluation from evaluator LLM."""
        evaluation_prompt = self._create_evaluation_prompt(question, response)
        evaluation_response = self.evaluator.generate(evaluation_prompt)
        
        try:
            evaluation = json.loads(evaluation_response)
            required_keys = {'score', 'feedback', 'strengths', 'areas_for_improvement'}
            if not all(key in evaluation for key in required_keys):
                raise ValueError("Missing required keys in evaluation response")
            return evaluation
        except json.JSONDecodeError:
            # Fallback if response isn't valid JSON
            return {
                "score": 0,
                "feedback": "Error parsing evaluation response",
                "strengths": [],
                "areas_for_improvement": ["Invalid response format"]
            }

    def _log_iteration(
        self,
        iteration: int,
        question: str,
        response: str,
        evaluation: Dict
    ):
        """Log iteration details to file."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "iteration": iteration,
            "question": question,
            "response": response,
            "evaluation": evaluation
        }
        
        try:
            with open(self.log_file, 'r') as f:
                logs = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            logs = []
            
        logs.append(log_entry)
        
        with open(self.log_file, 'w') as f:
            json.dump(logs, f, indent=2)

    def generate(
        self,
        prompt: str,
        verbose: bool = False
    ) -> Tuple[str, Dict, int]:
        """
        Generate response with iterative improvement.
        
        Parameters:
        - prompt (str): Input prompt/question
        - verbose (bool): Whether to print iteration details
        
        Returns:
        - Tuple[str, Dict, int]: (final_response, final_evaluation, iterations)
        """
        iteration = 0
        current_response = self.generator.generate(prompt)
        
        while iteration < self.max_iterations:
            iteration += 1
            
            # Evaluate current response
            evaluation = self._evaluate_response(prompt, current_response)
            
            # Log iteration
            self._log_iteration(iteration, prompt, current_response, evaluation)
            
            if verbose:
                print(f"\nIteration {iteration}:")
                print(f"Response: {current_response}")
                print(f"Score: {evaluation['score']}")
                print(f"Feedback: {evaluation['feedback']}")
            
            # Check if we've reached the threshold
            if evaluation['score'] >= self.threshold:
                if verbose:
                    print(f"\nReached threshold score of {self.threshold}")
                return current_response, evaluation, iteration
            
            # Generate improved response
            improvement_prompt = self._create_improvement_prompt(
                prompt,
                current_response,
                evaluation
            )
            current_response = self.generator.generate(improvement_prompt)
        
        if verbose:
            print(f"\nReached maximum iterations ({self.max_iterations})")
        
        # Final evaluation
        final_evaluation = self._evaluate_response(prompt, current_response)
        return current_response, final_evaluation, iteration

