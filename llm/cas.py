# llm/cas.py

from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
import json
from .llm import LLM

@dataclass
class Step:
    """Represents a single step in the chain."""
    number: int
    description: str
    status: str  # 'pending', 'in_progress', 'completed', 'failed'
    input: str
    output: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict:
        """Convert step to dictionary format."""
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
    def from_dict(cls, data: Dict) -> 'Step':
        """Create step from dictionary format."""
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

class ChainAutoStep:
    def __init__(
        self,
        llm: LLM,
        history_file: str = "cas_history.json",
        verbose: bool = False
    ):
        """
        Initialize Chain Auto Step processor.
        
        Parameters:
        - llm (LLM): LLM instance for processing
        - history_file (str): File to store execution history
        - verbose (bool): Whether to print detailed execution information
        """
        self.llm = llm
        self.history_file = history_file
        self.verbose = verbose
        self.current_execution: Optional[Dict] = None
        
    def _create_planning_prompt(self, query: str) -> str:
        """Create prompt for planning steps."""
        return f"""Break down the following query into detailed sequential steps.
        Format your response as a JSON array of steps, where each step has:
        - "number": Step number
        - "description": Detailed description of what needs to be done
        - "input": What information is needed for this step
        
        Query: {query}
        
        Example format:
        [
            {{
                "number": 1,
                "description": "First step description",
                "input": "Information needed for step 1"
            }},
            ...
        ]
        
        Ensure steps are:
        1. Sequential and dependent on previous steps when necessary
        2. Specific and actionable
        3. Clear about input requirements
        
        Please provide the steps in JSON format:"""

    def _create_execution_prompt(self, step: Step, previous_outputs: List[str]) -> str:
        """Create prompt for executing a specific step."""
        context = "\n".join([
            f"Step {i+1} output: {output}"
            for i, output in enumerate(previous_outputs)
        ])
        
        return f"""Execute the following step based on the context provided.
        
        Previous steps context:
        {context}
        
        Current step to execute:
        Number: {step.number}
        Description: {step.description}
        Input required: {step.input}
        
        Please provide the result for this specific step:"""

    def _parse_steps(self, planning_response: str) -> List[Step]:
        """Parse LLM response into list of steps."""
        try:
            # Try to find and parse JSON in the response
            start = planning_response.find('[')
            end = planning_response.rfind(']') + 1
            if start == -1 or end == 0:
                raise ValueError("No JSON array found in response")
            
            steps_data = json.loads(planning_response[start:end])
            steps = []
            
            for step_data in steps_data:
                step = Step(
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
            
        # Update or append current execution
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
        Execute a query through automated steps.
        
        Parameters:
        - query (str): The query to process
        
        Returns:
        - Dict: Execution results including all steps and final output
        """
        # Initialize new execution
        self.current_execution = {
            "execution_id": datetime.now().isoformat(),
            "query": query,
            "started_at": datetime.now().isoformat(),
            "status": "in_progress",
            "steps": [],
            "final_output": None
        }
        
        try:
            # Plan steps
            if self.verbose:
                print("Planning steps...")
            
            planning_prompt = self._create_planning_prompt(query)
            planning_response = self.llm.generate(planning_prompt)
            steps = self._parse_steps(planning_response)
            
            if self.verbose:
                print(f"Planned {len(steps)} steps:")
                for step in steps:
                    print(f"Step {step.number}: {step.description}")
            
            # Execute steps sequentially
            previous_outputs = []
            final_output = None
            
            for step in steps:
                if self.verbose:
                    print(f"\nExecuting step {step.number}...")
                
                step.status = "in_progress"
                step.started_at = datetime.now()
                
                # Create execution prompt with context from previous steps
                execution_prompt = self._create_execution_prompt(step, previous_outputs)
                
                # Execute step
                try:
                    step_output = self.llm.generate(execution_prompt)
                    step.output = step_output
                    step.status = "completed"
                    previous_outputs.append(step_output)
                    final_output = step_output  # Last step output becomes final output
                    
                except Exception as e:
                    step.status = "failed"
                    step.output = f"Error: {str(e)}"
                    raise
                
                finally:
                    step.completed_at = datetime.now()
                    
                # Update execution record
                self.current_execution["steps"].append(step.to_dict())
                self._save_execution()
            
            # Complete execution
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
