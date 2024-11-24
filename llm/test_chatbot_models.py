# test_chatbot_models.py

from llm.llm import LLM
from llm.chatbot import Chatbot
import time
from datetime import datetime
import json
import os
import sys
from typing import Dict, List, Optional
from tqdm import tqdm
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

@dataclass
class TestScenario:
    """Represents a test scenario for the chatbot."""
    name: str
    description: str
    system_prompt: str
    context: Optional[str]
    entities: Dict
    conversation: List[str]
    expected_patterns: List[str]
    category: str

class ChatbotTester:
    def __init__(self):
        # Create base directory for all test outputs
        self.base_dir = "chatbot_tests"
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
        
        self.test_scenarios = [
            TestScenario(
                name="Basic_Conversation",
                description="Test basic conversational capabilities",
                system_prompt="You are a helpful assistant.",
                context=None,
                entities={},
                conversation=[
                    "Hello, how are you?",
                    "What's the weather like?",
                    "Tell me a joke."
                ],
                expected_patterns=[
                    "greeting",
                    "weather",
                    "humor"
                ],
                category="general"
            ),
            TestScenario(
                name="E_commerce_Support",
                description="Test product-related queries and support",
                system_prompt="You are a customer service representative for an electronics store.",
                context="You work for TechStore, a premium electronics retailer.",
                entities={
                    "iphone_13": {
                        "name": "iPhone 13",
                        "price": 799,
                        "stock": 50,
                        "colors": ["black", "white", "blue"],
                        "storage": ["128GB", "256GB", "512GB"]
                    },
                    "samsung_s21": {
                        "name": "Samsung Galaxy S21",
                        "price": 699,
                        "stock": 30,
                        "colors": ["phantom gray", "phantom white"],
                        "storage": ["128GB", "256GB"]
                    }
                },
                conversation=[
                    "I'm looking for a new smartphone",
                    "What's the price of the iPhone 13?",
                    "Do you have it in blue?",
                    "How does it compare to the Samsung S21?"
                ],
                expected_patterns=[
                    "product_inquiry",
                    "price_info",
                    "availability",
                    "comparison"
                ],
                category="customer_service"
            ),
            TestScenario(
                name="Technical_Support",
                description="Test technical troubleshooting capabilities",
                system_prompt="You are a technical support specialist.",
                context="You provide support for computer hardware and software issues.",
                entities={
                    "windows_update": {
                        "name": "Windows Update",
                        "current_version": "21H2",
                        "known_issues": ["slow installation", "driver conflicts"]
                    },
                    "antivirus": {
                        "name": "Security Suite Pro",
                        "version": "2024.1",
                        "features": ["real-time protection", "firewall", "ransomware protection"]
                    }
                },
                conversation=[
                    "My computer is running slowly",
                    "Should I update Windows?",
                    "What about antivirus software?",
                    "How do I check for malware?"
                ],
                expected_patterns=[
                    "problem_diagnosis",
                    "update_recommendation",
                    "security_advice",
                    "troubleshooting"
                ],
                category="technical"
            ),
            TestScenario(
                name="Complex_Problem_Solving",
                description="Test ability to handle complex scenarios",
                system_prompt="You are an AI consultant specializing in problem-solving.",
                context="You help clients solve complex business and technical challenges.",
                entities={
                    "project_management": {
                        "name": "Agile Framework",
                        "methodologies": ["Scrum", "Kanban"],
                        "tools": ["Jira", "Trello"]
                    },
                    "data_analysis": {
                        "name": "Analytics Suite",
                        "tools": ["Python", "R", "SQL"],
                        "techniques": ["regression", "clustering"]
                    }
                },
                conversation=[
                    "How can I improve my team's productivity?",
                    "What project management methodology would you recommend?",
                    "How can we better analyze our data?",
                    "Can you suggest some automation strategies?"
                ],
                expected_patterns=[
                    "analysis",
                    "recommendation",
                    "methodology",
                    "strategy"
                ],
                category="consulting"
            )
        ]
    def analyze_response(self, response: str, expected_pattern: str) -> Dict:
        """Analyze response characteristics and pattern matching."""
        analysis = {
            "length": len(response),
            "word_count": len(response.split()),
            "sentence_count": len(response.split('.')),
            "has_question": '?' in response,
            "has_entities": any(entity in response.lower() for entity in ['iphone', 'samsung', 'windows', 'security']),
            "formality_score": self.calculate_formality_score(response),
            "pattern_match": self.check_pattern_match(response, expected_pattern)
        }
        return analysis

    def calculate_formality_score(self, text: str) -> float:
        """Calculate a simple formality score based on language patterns."""
        formal_indicators = [
            'please', 'would', 'could', 'thank you', 'appreciate',
            'recommend', 'suggest', 'advise', 'consider'
        ]
        informal_indicators = [
            'hey', 'hi', 'okay', 'ok', 'yeah', 'nope', 'gonna', 'wanna'
        ]
        
        formal_count = sum(text.lower().count(word) for word in formal_indicators)
        informal_count = sum(text.lower().count(word) for word in informal_indicators)
        
        total = formal_count + informal_count
        if total == 0:
            return 0.5
        return formal_count / total

    def check_pattern_match(self, response: str, expected_pattern: str) -> Dict:
        """Check if response matches expected pattern."""
        patterns = {
            "greeting": ['hello', 'hi', 'greetings', 'welcome'],
            "weather": ['weather', 'temperature', 'forecast', 'climate'],
            "humor": ['joke', 'funny', 'laugh', 'humor'],
            "product_inquiry": ['features', 'specifications', 'details', 'available'],
            "price_info": ['price', 'cost', 'dollars', '$'],
            "availability": ['available', 'in stock', 'inventory', 'supply'],
            "comparison": ['compared', 'versus', 'better', 'difference'],
            "problem_diagnosis": ['issue', 'problem', 'cause', 'symptom'],
            "update_recommendation": ['update', 'upgrade', 'latest version', 'patch'],
            "security_advice": ['security', 'protection', 'safeguard', 'defend'],
            "troubleshooting": ['troubleshoot', 'fix', 'solve', 'resolve'],
            "analysis": ['analyze', 'evaluate', 'assess', 'measure'],
            "recommendation": ['recommend', 'suggest', 'advise', 'consider'],
            "methodology": ['method', 'approach', 'process', 'framework'],
            "strategy": ['strategy', 'plan', 'roadmap', 'initiative']
        }
        
        if expected_pattern not in patterns:
            return {"matched": False, "confidence": 0.0}
        
        pattern_words = patterns[expected_pattern]
        matches = sum(word in response.lower() for word in pattern_words)
        confidence = matches / len(pattern_words)
        
        return {
            "matched": confidence > 0.3,
            "confidence": confidence
        }

    def analyze_conversation_flow(self, conversation_results: List[Dict]) -> Dict:
        """Analyze the flow and coherence of the conversation."""
        return {
            "response_times": [r["duration"] for r in conversation_results],
            "avg_response_time": sum(r["duration"] for r in conversation_results) / len(conversation_results),
            "pattern_match_rates": [r["analysis"]["pattern_match"]["confidence"] for r in conversation_results],
            "avg_pattern_match": sum(r["analysis"]["pattern_match"]["confidence"] for r in conversation_results) / len(conversation_results),
            "formality_consistency": self._calculate_formality_consistency([r["analysis"]["formality_score"] for r in conversation_results])
        }

    def _calculate_formality_consistency(self, formality_scores: List[float]) -> float:
        """Calculate consistency in formality across responses."""
        if not formality_scores:
            return 0.0
        avg = sum(formality_scores) / len(formality_scores)
        variance = sum((score - avg) ** 2 for score in formality_scores) / len(formality_scores)
        return 1.0 - min(1.0, variance)  # Convert to consistency score (1 = very consistent)

    def test_chatbot(
        self,
        provider: str,
        model: str,
        scenario: TestScenario,
        max_retries: int = 3
    ) -> Dict:
        """Test a specific chatbot configuration with a scenario."""
        for attempt in range(max_retries):
            try:
                # Initialize LLM
                llm = LLM(
                    provider=provider,
                    model=model,
                    temperature=0.7
                )
                
                # Create safe filenames
                safe_name = "".join(c for c in scenario.name if c.isalnum() or c in ('-', '_'))
                history_file = os.path.join(
                    self.results_dir,
                    f"history_{provider}_{model}_{safe_name}.json"
                )
                entities_file = os.path.join(
                    self.results_dir,
                    f"entities_{provider}_{model}_{safe_name}.json"
                )
                
                # Initialize Chatbot
                chatbot = Chatbot(
                    llm=llm,
                    system_prompt=scenario.system_prompt,
                    context=scenario.context,
                    history_file=history_file,
                    entities_file=entities_file
                )
                
                # Add entities
                for entity_name, entity_data in scenario.entities.items():
                    chatbot.add_entity(entity_name, entity_data)
                
                # Run conversation
                conversation_results = []
                total_duration = 0
                
                for query, expected_pattern in zip(scenario.conversation, scenario.expected_patterns):
                    start_time = time.time()
                    response = chatbot.chat(query)
                    duration = time.time() - start_time
                    
                    # Analyze response
                    analysis = self.analyze_response(response, expected_pattern)
                    
                    conversation_results.append({
                        "query": query,
                        "response": response,
                        "duration": duration,
                        "expected_pattern": expected_pattern,
                        "analysis": analysis
                    })
                    
                    total_duration += duration
                
                # Analyze conversation flow
                flow_analysis = self.analyze_conversation_flow(conversation_results)
                
                return {
                    "success": True,
                    "provider": provider,
                    "model": model,
                    "scenario": scenario.name,
                    "category": scenario.category,
                    "conversation": conversation_results,
                    "flow_analysis": flow_analysis,
                    "total_duration": total_duration,
                    "average_duration": total_duration / len(scenario.conversation),
                    "attempt": attempt + 1,
                    "error": None
                }
                
            except Exception as e:
                if attempt == max_retries - 1:
                    return {
                        "success": False,
                        "provider": provider,
                        "model": model,
                        "scenario": scenario.name,
                        "category": scenario.category,
                        "conversation": None,
                        "flow_analysis": None,
                        "total_duration": None,
                        "average_duration": None,
                        "attempt": attempt + 1,
                        "error": str(e)
                    }
                time.sleep(2 ** attempt)  # Exponential backoff
    def run_tests(self, max_workers: int = 4) -> List[Dict]:
        """Run all test scenarios for all models."""
        results = []
        total_tests = len(self.test_cases) * len(self.test_scenarios)
        
        with tqdm(total=total_tests, desc="Running tests") as pbar:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_test = {
                    executor.submit(
                        self.test_chatbot,
                        test_case["provider"],
                        test_case["model"],
                        scenario
                    ): (test_case, scenario)
                    for test_case in self.test_cases
                    for scenario in self.test_scenarios
                }
                
                for future in future_to_test:
                    test_case, scenario = future_to_test[future]
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        results.append({
                            "success": False,
                            "provider": test_case["provider"],
                            "model": test_case["model"],
                            "scenario": scenario.name,
                            "error": str(e)
                        })
                    finally:
                        pbar.update(1)
                        
                        # Print progress
                        if result["success"]:
                            tqdm.write(
                                f"✅ {result['provider']} - {result['model']} - "
                                f"{result['scenario']} ({result['average_duration']:.2f}s/response)"
                            )
                        else:
                            tqdm.write(
                                f"❌ {result['provider']} - {result['model']} - "
                                f"{result['scenario']}"
                            )
                            tqdm.write(f"   Error: {result['error']}")
        
        return results

    def save_results(self, results: List[Dict], timestamp: str) -> str:
        """Save test results to a JSON file."""
        filename = os.path.join(
            self.results_dir,
            f"chatbot_test_results_{timestamp}.json"
        )
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
            
        return filename

    def save_report(self, report: Dict, timestamp: str) -> str:
        """Save test report to a JSON file."""
        filename = os.path.join(
            self.reports_dir,
            f"chatbot_test_report_{timestamp}.json"
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
            "by_scenario": {},
            "performance_metrics": {}
        }
        
        # Analyze by provider and model
        for result in results:
            if not result["success"]:
                continue
                
            provider = result["provider"]
            model = result["model"]
            key = f"{provider}/{model}"
            
            if key not in report["by_provider"]:
                report["by_provider"][key] = {
                    "total_conversations": 0,
                    "avg_duration": 0,
                    "avg_word_count": 0,
                    "avg_formality": 0,
                    "pattern_match_rate": 0,
                    "scenarios_completed": []
                }
            
            stats = report["by_provider"][key]
            stats["total_conversations"] += len(result["conversation"])
            stats["avg_duration"] += result["average_duration"]
            stats["scenarios_completed"].append(result["scenario"])
            
            for conv in result["conversation"]:
                stats["avg_word_count"] += conv["analysis"]["word_count"]
                stats["avg_formality"] += conv["analysis"]["formality_score"]
                stats["pattern_match_rate"] += conv["analysis"]["pattern_match"]["confidence"]
        
        # Calculate averages
        for stats in report["by_provider"].values():
            total = stats["total_conversations"]
            if total > 0:
                stats["avg_duration"] /= len(stats["scenarios_completed"])
                stats["avg_word_count"] /= total
                stats["avg_formality"] /= total
                stats["pattern_match_rate"] /= total
        
        return report

    def print_report(self, report: Dict):
        """Print formatted test report."""
        print("\n=== Chatbot Test Report ===")
        print(f"\nTotal Tests: {report['summary']['total_tests']}")
        print(f"Successful: {report['summary']['successful_tests']}")
        print(f"Failed: {report['summary']['failed_tests']}")
        
        print("\n=== Provider Performance ===")
        for provider, stats in report["by_provider"].items():
            print(f"\n{provider}:")
            print(f"Total Conversations: {stats['total_conversations']}")
            print(f"Average Duration: {stats['avg_duration']:.2f}s")
            print(f"Average Word Count: {stats['avg_word_count']:.1f}")
            print(f"Average Formality: {stats['avg_formality']:.2f}")
            print(f"Pattern Match Rate: {stats['pattern_match_rate']*100:.1f}%")

def main():
    try:
        # Initialize tester
        tester = ChatbotTester()
        
        # Run tests
        print("Starting Chatbot tests...")
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
        print(f"Report saved to {report_file}")
        
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
