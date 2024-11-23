# llm/cds.py

from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
import json
from .llm import LLM

@dataclass
class DeterminedStep:
    """Represents a single step in the determined chain."""
    number: int
    description: str
    status: str
    input: str
    output: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict:
        return {
            "number": self.number,
            "description": self.description,
            "status": self.status,
            "input": self.input,
            "output": self.output,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'DeterminedStep':
        step = cls(
            number=data["number"],
            description=data["description"],
            status=data["status"],
            input=data["input"],
            output=data["output"]
        )
        if data["started_at"]:
            step.started_at = datetime.fromisoformat(data["started_at"])
        if data["completed_at"]:
            step.completed_at = datetime.fromisoformat(data["completed_at"])
        return step

class ChainDeterminedStep:
    def __init__(
        self,
        llm: LLM,
        num_steps: int,
        history_file: str = "cds_history.json",
        verbose: bool = False
    ):
        """
        Initialize Chain Determined Step processor.
        
        Parameters:
        - llm (LLM): LLM instance for processing
        - num_steps (int): Number of steps to break the task into
        - history_file (str): File to store execution history
        - verbose (bool): Whether to print detailed execution information
        """
        if num_steps < 1:
            raise ValueError("Number of steps must be at least 1")
            
        self.llm = llm
        self.num_steps = num_steps
        self.history_file = history_file
        self.verbose = verbose
        self.current_execution: Optional[Dict] = None

    def _create_planning_prompt(self, query: str) -> str:
        """Create prompt for planning exact number of steps."""
        return f"""Break down the following query into exactly {self.num_steps} sequential steps.
        Each step should be meaningful and contribute to the final goal.
        
        Format your response as a JSON array of exactly {self.num_steps} steps, where each step has:
        - "number": Step number (1 to {self.num_steps})
        - "description": Detailed description of what needs to be done
        - "input": What information is needed for this step
        
        Query: {query}
        
        Requirements:
        1. Exactly {self.num_steps} steps
        2. Steps must be sequential and build upon each other
        3. Each step must be specific and actionable
        4. Steps should be of roughly equal complexity
        
        Please provide exactly {self.num_steps} steps in JSON format:"""

    def _create_execution_prompt(self, step: DeterminedStep, previous_outputs: List[str]) -> str:
        """Create prompt for executing a specific step."""
        context = "\n".join([
            f"Step {i+1} output: {output}"
            for i, output in enumerate(previous_outputs)
        ])
        
        return f"""Execute step {step.number} of {self.num_steps} based on the context provided.
        
        Previous steps context:
        {context}
        
        Current step ({step.number}/{self.num_steps}):
        Description: {step.description}
        Input required: {step.input}
        
        Please provide the result for this specific step, keeping in mind this is step {step.number} of {self.num_steps}:"""

    def _parse_steps(self, planning_response: str) -> List[DeterminedStep]:
        """Parse LLM response into exactly num_steps steps."""
        try:
            start = planning_response.find('[')
            end = planning_response.rfind(']') + 1
            if start == -1 or end == 0:
                raise ValueError("No JSON array found in response")
            
            steps_data = json.loads(planning_response[start:end])
            
            if len(steps_data) != self.num_steps:
                raise ValueError(f"Expected exactly {self.num_steps} steps, got {len(steps_data)}")
            
            steps = []
            for step_data in steps_data:
                step = DeterminedStep(
                    number=step_data["number"],
                    description=step_data["description"],
                    status="pending",
                    input=step_data["input"]
                )
                steps.append(step)
            
            return steps
            
        except (json.JSONDecodeError, KeyError) as e:
            raise ValueError(f"Error parsing steps: {e}")

    def _save_execution(self):
        """Save current execution to history file."""
        if not self.current_execution:
            return
            
        try:
            with open(self.history_file, 'r') as f:
                history = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            history = []
            
        execution_id = self.current_execution["execution_id"]
        for i, exec in enumerate(history):
            if exec["execution_id"] == execution_id:
                history[i] = self.current_execution
                break
        else:
            history.append(self.current_execution)
            
        with open(self.history_file, 'w') as f:
            json.dump(history, f, indent=2)

    def execute(self, query: str) -> Dict:
        """
        Execute a query through determined number of steps.
        
        Parameters:
        - query (str): The query to process
        
        Returns:
        - Dict: Execution results including all steps and final output
        """
        self.current_execution = {
            "execution_id": datetime.now().isoformat(),
            "query": query,
            "num_steps": self.num_steps,
            "started_at": datetime.now().isoformat(),
            "status": "in_progress",
            "steps": [],
            "final_output": None
        }
        
        try:
            if self.verbose:
                print(f"Planning {self.num_steps} steps...")
            
            planning_prompt = self._create_planning_prompt(query)
            planning_response = self.llm.generate(planning_prompt)
            steps = self._parse_steps(planning_response)
            
            if self.verbose:
                print(f"\nPlanned {self.num_steps} steps:")
                for step in steps:
                    print(f"Step {step.number}/{self.num_steps}: {step.description}")
            
            previous_outputs = []
            final_output = None
            
            for step in steps:
                if self.verbose:
                    print(f"\nExecuting step {step.number}/{self.num_steps}...")
                
                step.status = "in_progress"
                step.started_at = datetime.now()
                
                execution_prompt = self._create_execution_prompt(step, previous_outputs)
                
                try:
                    step_output = self.llm.generate(execution_prompt)
                    step.output = step_output
                    step.status = "completed"
                    previous_outputs.append(step_output)
                    final_output = step_output
                    
                except Exception as e:
                    step.status = "failed"
                    step.output = f"Error: {str(e)}"
                    raise
                
                finally:
                    step.completed_at = datetime.now()
                    
                self.current_execution["steps"].append(step.to_dict())
                self._save_execution()
            
            self.current_execution["status"] = "completed"
            self.current_execution["completed_at"] = datetime.now().isoformat()
            self.current_execution["final_output"] = final_output
            
        except Exception as e:
            self.current_execution["status"] = "failed"
            self.current_execution["error"] = str(e)
            self.current_execution["completed_at"] = datetime.now().isoformat()
            raise
        
        finally:
            self._save_execution()
            
        return self.current_execution
    
    def get_execution_history(self) -> List[Dict]:
        """Get all previous executions."""
        try:
            with open(self.history_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []
    
    def get_execution(self, execution_id: str) -> Optional[Dict]:
        """Get specific execution by ID."""
        history = self.get_execution_history()
        for execution in history:
            if execution["execution_id"] == execution_id:
                return execution
        return None

