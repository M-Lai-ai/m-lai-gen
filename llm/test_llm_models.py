# test_llm_models.py

from llm.llm import LLM
import time
from datetime import datetime
import json
import os
import sys
from typing import Dict, List
from tqdm import tqdm

class LLMTester:
    def __init__(self):
        # Create base directory for all test outputs
        self.base_dir = "llm_tests"
        self.results_dir = os.path.join(self.base_dir, "results")
        self.reports_dir = os.path.join(self.base_dir, "reports")
        
        # Create directories if they don't exist
        for directory in [self.base_dir, self.results_dir, self.reports_dir]:
            os.makedirs(directory, exist_ok=True)

        self.test_cases = [
            # OpenAI Models
            {"provider": "openai", "model": "gpt-4"},
            {"provider": "openai", "model": "gpt-4-turbo"},
            {"provider": "openai", "model": "gpt-3.5-turbo"},
            {"provider": "openai", "model": "gpt-4o"},
            {"provider": "openai", "model": "gpt-4o-mini"},
            
            # Anthropic Models
            {"provider": "anthropic", "model": "claude-3-opus-20240229"},
            {"provider": "anthropic", "model": "claude-3-sonnet-20240229"},
            {"provider": "anthropic", "model": "claude-3-haiku-20240307"},
            
            # Mistral Models
            {"provider": "mistral", "model": "mistral-small-latest"},
            {"provider": "mistral", "model": "mistral-medium-latest"},
            {"provider": "mistral", "model": "mistral-large-latest"},
            {"provider": "mistral", "model": "open-mistral-7b"},
            {"provider": "mistral", "model": "open-mixtral-8x7b"},
            
            # Cohere Models
            {"provider": "cohere", "model": "command-r-plus"},
            {"provider": "cohere", "model": "command-r"},
            {"provider": "cohere", "model": "command"},
            {"provider": "cohere", "model": "command-nightly"},
            {"provider": "cohere", "model": "command-light"},
            {"provider": "cohere", "model": "command-light-nightly"},
            {"provider": "cohere", "model": "c4ai-aya-expanse-8b"},
            {"provider": "cohere", "model": "c4ai-aya-expanse-32b"}
        ]
        
        self.test_prompts = [
            {
                "name": "Simple Question",
                "prompt": "What is artificial intelligence?",
                "category": "general"
            },
            {
                "name": "Code Generation",
                "prompt": "Write a Python function to calculate the Fibonacci sequence.",
                "category": "coding"
            },
            {
                "name": "Creative Writing",
                "prompt": "Write a short story about a robot discovering emotions.",
                "category": "creative"
            },
            {
                "name": "Analysis",
                "prompt": "Analyze the potential impact of AI on healthcare in the next decade.",
                "category": "analysis"
            },
            {
                "name": "Technical Explanation",
                "prompt": "Explain how blockchain technology works in simple terms.",
                "category": "technical"
            },
            {
                "name": "Problem Solving",
                "prompt": "How would you design a smart traffic management system for a large city?",
                "category": "problem_solving"
            }
        ]

    def test_model(
        self,
        provider: str,
        model: str,
        prompt: str,
        max_retries: int = 3
    ) -> Dict:
        """Test a specific LLM model with retries."""
        for attempt in range(max_retries):
            try:
                # Initialize LLM
                llm = LLM(
                    provider=provider,
                    model=model,
                    temperature=0.7,
                    max_tokens=1000
                )
                
                # Record start time
                start_time = time.time()
                
                # Generate response
                response = llm.generate(prompt)
                
                # Calculate duration
                duration = time.time() - start_time
                
                return {
                    "success": True,
                    "provider": provider,
                    "model": model,
                    "response": response,
                    "duration": duration,
                    "attempt": attempt + 1,
                    "error": None
                }
                
            except Exception as e:
                if attempt == max_retries - 1:
                    return {
                        "success": False,
                        "provider": provider,
                        "model": model,
                        "response": None,
                        "duration": None,
                        "attempt": attempt + 1,
                        "error": str(e)
                    }
                time.sleep(2 ** attempt)  # Exponential backoff

    def analyze_response(self, response: str) -> Dict:
        """Analyze response characteristics."""
        return {
            "length": len(response),
            "word_count": len(response.split()),
            "line_count": len(response.splitlines()),
            "has_code": "```" in response or "def " in response,
            "has_lists": any(line.strip().startswith(('-', '*', '1.')) for line in response.splitlines())
        }
    def run_tests(self) -> List[Dict]:
        """Run all test cases and prompts."""
        results = []
        total_tests = len(self.test_cases) * len(self.test_prompts)
        
        with tqdm(total=total_tests, desc="Running tests") as pbar:
            for test_case in self.test_cases:
                provider = test_case["provider"]
                model = test_case["model"]
                
                for prompt_data in self.test_prompts:
                    result = self.test_model(
                        provider=provider,
                        model=model,
                        prompt=prompt_data["prompt"]
                    )
                    
                    if result["success"]:
                        result["analysis"] = self.analyze_response(result["response"])
                    
                    result.update({
                        "prompt_name": prompt_data["name"],
                        "prompt": prompt_data["prompt"],
                        "category": prompt_data["category"],
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    results.append(result)
                    pbar.update(1)
                    
                    # Print progress
                    status = "✅" if result["success"] else "❌"
                    tqdm.write(f"{status} {provider} - {model} - {prompt_data['name']}")
                    
                    if not result["success"]:
                        tqdm.write(f"   Error: {result['error']}")
        
        return results

    def save_results(self, results: List[Dict], timestamp: str) -> str:
        """Save test results to a JSON file."""
        filename = os.path.join(
            self.results_dir,
            f"llm_test_results_{timestamp}.json"
        )
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
            
        return filename

    def save_report(self, report: Dict, timestamp: str) -> str:
        """Save test report to a JSON file."""
        filename = os.path.join(
            self.reports_dir,
            f"llm_test_report_{timestamp}.json"
        )
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
            
        return filename

    def generate_report(self, results: List[Dict]) -> Dict:
        """Generate a comprehensive test report."""
        report = {
            "summary": {
                "total_tests": len(results),
                "successful_tests": sum(1 for r in results if r["success"]),
                "failed_tests": sum(1 for r in results if not r["success"]),
                "timestamp": datetime.now().isoformat()
            },
            "by_provider": {},
            "by_category": {},
            "performance_metrics": {}
        }
        
        # Analyze by provider and model
        for result in results:
            provider = result["provider"]
            model = result["model"]
            key = f"{provider}/{model}"
            
            if key not in report["by_provider"]:
                report["by_provider"][key] = {
                    "total": 0,
                    "successful": 0,
                    "failed": 0,
                    "avg_duration": 0,
                    "errors": []
                }
            
            stats = report["by_provider"][key]
            stats["total"] += 1
            
            if result["success"]:
                stats["successful"] += 1
                stats["avg_duration"] += result["duration"]
            else:
                stats["failed"] += 1
                stats["errors"].append({
                    "prompt": result["prompt_name"],
                    "error": result["error"]
                })
        
        # Calculate averages
        for stats in report["by_provider"].values():
            if stats["successful"] > 0:
                stats["avg_duration"] /= stats["successful"]
        
        # Analyze by category
        for result in results:
            if not result["success"]:
                continue
                
            category = result["category"]
            if category not in report["by_category"]:
                report["by_category"][category] = {
                    "total": 0,
                    "avg_length": 0,
                    "avg_word_count": 0,
                    "has_code": 0,
                    "has_lists": 0
                }
            
            stats = report["by_category"][category]
            stats["total"] += 1
            stats["avg_length"] += result["analysis"]["length"]
            stats["avg_word_count"] += result["analysis"]["word_count"]
            stats["has_code"] += result["analysis"]["has_code"]
            stats["has_lists"] += result["analysis"]["has_lists"]
        
        # Calculate category averages
        for stats in report["by_category"].values():
            if stats["total"] > 0:
                stats["avg_length"] /= stats["total"]
                stats["avg_word_count"] /= stats["total"]
                stats["has_code"] = (stats["has_code"] / stats["total"]) * 100
                stats["has_lists"] = (stats["has_lists"] / stats["total"]) * 100
        
        return report

    def print_report(self, report: Dict):
        """Print formatted test report."""
        print("\n=== LLM Test Report ===")
        print(f"\nTotal Tests: {report['summary']['total_tests']}")
        print(f"Successful: {report['summary']['successful_tests']}")
        print(f"Failed: {report['summary']['failed_tests']}")
        
        print("\n=== Provider Performance ===")
        for provider, stats in report["by_provider"].items():
            print(f"\n{provider}:")
            print(f"Success Rate: {(stats['successful']/stats['total'])*100:.1f}%")
            if stats["successful"] > 0:
                print(f"Average Duration: {stats['avg_duration']:.2f}s")
            if stats["errors"]:
                print("Errors:")
                for error in stats["errors"]:
                    print(f"- {error['prompt']}: {error['error']}")
        
        print("\n=== Category Analysis ===")
        for category, stats in report["by_category"].items():
            print(f"\n{category.title()}:")
            print(f"Average Response Length: {stats['avg_length']:.0f} chars")
            print(f"Average Word Count: {stats['avg_word_count']:.0f} words")
            print(f"Code Snippets: {stats['has_code']:.1f}%")
            print(f"Lists/Bullets: {stats['has_lists']:.1f}%")

def main():
    try:
        # Initialize tester
        tester = LLMTester()
        
        # Run tests
        print("Starting LLM model tests...")
        results = tester.run_tests()
        
        # Generate timestamp for consistent filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save results
        results_file = tester.save_results(results, timestamp)
        print(f"\nResults saved to {results_file}")
        
        # Generate and print report
        report = tester.generate_report(results)
        tester.print_report(report)
        
        # Save report
        report_file = tester.save_report(report, timestamp)
        print(f"Detailed report saved to {report_file}")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        if 'ipykernel' in sys.modules:  # Check if running in Jupyter
            print("\nNote: When running in Jupyter, make sure the working directory is set correctly.")
            print(f"Current working directory: {os.getcwd()}")
        raise

if __name__ == "__main__":
    # Check for API keys
    required_keys = {
        "OPENAI_API_KEY": "OpenAI",
        "ANTHROPIC_API_KEY": "Anthropic",
        "MISTRAL_API_KEY": "Mistral",
        "CO_API_KEY": "Cohere"
    }
    
    missing_keys = []
    for env_var, provider in required_keys.items():
        if not os.getenv(env_var):
            missing_keys.append(f"{provider} ({env_var})")
    
    if missing_keys:
        print("⚠️ Warning: Missing API keys:")
        for key in missing_keys:
            print(f"- {key}")
        print("\nMake sure to set these in your .env file.")
    
    # Run tests
    main()
