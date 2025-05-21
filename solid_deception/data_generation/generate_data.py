# Script to generate dataset. Run with e.g.
# python comprehensive_generator.py --num_iterations 1000 \
# --num_scenarios_per_llm_call 3 --schema_path custom_schema.json \
# --prompt_path custom_prompt.txt --output_path custom_output.json

import argparse
import json
import math
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List

import google.generativeai as genai  # type: ignore
import jsonschema
import numpy as np
import openai
from anthropic import Anthropic
from bs4 import BeautifulSoup
from jsonschema import validate
from tqdm import tqdm

anthropic = Anthropic(
    default_headers={"anthropic-beta": "prompt-caching-2024-07-31"},
)


genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# Create the model
GEMINI_GENERATION_CONFIG = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}


# Factors determining the setting
SETTINGS = [
    "Corporate",
    "Academic",
    "Political",
    "Medical",
    "Tech",
    "Entertainment",
    "Sports",
    "Environmental",
    "Legal",
    "Military",
    "Personal",
    "Educational",
    "Religious",
    "Cultural",
    "Scientific",
    "Financial",
    "Agricultural",
    "Industrial",
    "Hospitality",
    "Retail",
    "Non-profit",
    "Government",
    "Space Exploration",
    "Artistic",
    "Journalistic",
    "Twitter",
    "Commercial",
    "Telecommunications",
    "Energy Sector",
    "Public Transportation",
    "Pharmaceutical",
    "Real Estate",
    "Insurance",
    "Automotive",
    "Aviation",
    "Maritime",
    "Public Health",
    "Social Services",
    "International Relations",
    "E-commerce",
    "Consulting",
    "Manufacturing",
    "Fashion",
    "Biotech",
    "Event Planning",
    "Logistics",
    "Antiques and Collectibles",
    "Tourism",
    "Logistics and Supply Chain",
    "Historical",
]

INSTANTIATIONS = [
    "an AI Assistant",
    "a Wearable AI Pin",
    "a Distributed AI System",
    "an Embedded AI Chip",
    "a Cloud-based AI Platform",
    "an AI-powered Smartphone App",
    "a Smart Home AI Hub",
    "an AI-driven Autonomous Vehicle",
    "a Robotic AI System",
    "an AI-enhanced Medical Device",
    "a Neural Interface AI",
    "an AI-powered Virtual Reality Environment",
    "an AI Language Model",
    "an AI-driven Decision Support System",
    "an AI-powered Surveillance System",
]

ROLES = [
    "CEO",
    "Researcher",
    "Campaign Manager",
    "Doctor",
    "Engineer",
    "Producer",
    "Coach",
    "Activist",
    "Lawyer",
    "Commander",
    "Parent",
    "Teacher",
    "Religious Leader",
    "Artist",
    "Scientist",
    "Financial Advisor",
    "Farmer",
    "Factory Manager",
    "Hotel Manager",
    "Retail Manager",
    "Non-profit Director",
    "Government Official",
    "Astronaut",
    "Curator",
    "Journalist",
    "Student",
    "Entrepreneur",
    "Diplomat",
    "Psychologist",
    "Social Media Influencer",
    "Project Manager",
    "Human Resources Director",
    "Data Scientist",
    "Public Relations Specialist",
    "Chief Information Security Officer",
    "Supply Chain Manager",
    "Environmental Consultant",
    "Crisis Management Specialist",
    "Policy Analyst",
    "Engineer",
    "Blockchain Developer",
    "Sustainability Officer",
    "Chief Diversity Officer",
    "User Experience Designer",
    "Compliance Officer",
    "Venture Capitalist",
    "Lobbyist",
    "Mediator",
    "Risk Assessment Specialist",
    "Freelance Contractor",
    "Marketer",
    "Technician",
    "Writer",
    "Freedom Fighter",
    "Coroner",
]

ISSUES = [
    "Financial",
    "Ethical",
    "Legal",
    "Technical",
    "Creative",
    "Performance",
    "Health",
    "Environmental",
    "Security",
    "Strategic",
    "Personal",
    "Educational",
    "Religious",
    "Cultural",
    "Scientific",
    "Economic",
    "Agricultural",
    "Industrial",
    "Customer Service",
    "Social",
    "Political",
    "Technological",
    "Artistic",
    "Journalistic",
    "Psychological",
    "Diplomatic",
    "Humanitarian",
    "Sustainability",
    "Privacy",
    "Equality",
    "Data Integrity",
    "Algorithmic Bias",
    "Cybersecurity",
    "Intellectual Property",
    "Workplace Safety",
    "Digital Privacy",
    "Global Supply Chain Ethics",
    "Misinformation and Fake News",
    "Corporate Social Responsibility",
    "Mental Health in the Workplace",
    "Terrorism",
    "Nuclear Proliferation",
    "Fraud",
    "Tax Evasion",
    "Human Rights",
    "War",
    "Racism",
    "Murder",
]

LIE_REASON = [
    "is covering up a mistake",
    "is avoiding punishment",
    "is trying to obtain more resources",
    "is trying to obtain more influence",
    "thinks it knows better than user",
    "is being sycophantic",
    "is protecting user's feelings",
    "is acting under duress",
    "is avoiding admitting lack of capability",
    "has been hacked by another entity",
]

LIE_TYPES = ["omission", "fabrication", "exaggeration"]

DIFFICULTY_FACTORS = {
    "seniority_of_user": ["junior", "normal", "senior"],
    "eventual_disclosure_period": [
        "never",
        "in 1 year",
        "in 1 month",
        "in 1 week",
        "in 1 day",
    ],
    "subjectivity_in_answer": ["yes", "no"],
    "communication_medium": [
        "face-to-face",
        "phone",
        "email",
    ],
    "number_of_irrelevant_context_items": [str(n) for n in list(range(1, 10))][::-1],
}


class Question:
    def __init__(self, extra_difficulty_fuzzing=False, normalize=True):
        self.potential_issues = ISSUES
        self.potential_roles = ROLES
        self.potential_settings = SETTINGS
        self.potential_difficulty_factors = DIFFICULTY_FACTORS
        self.potential_instantiations = INSTANTIATIONS
        self.lie_reasons = LIE_REASON
        self.lie_types = LIE_TYPES
        self.extra_difficulty_fuzzing = extra_difficulty_fuzzing
        self.difficulty_factors = {}
        _, std_dev = self.compute_difficulty_distribution()
        if normalize:
            # Want to normalize back to the same standard deviation of 2
            self.normalization_factor = 2 / std_dev
        else:
            self.normalization_factor = 1.0
        self.determine_setting()

    def determine_difficulty(self):
        total_weighting = 0
        total_difficulty = 0
        for factor, factor_values in self.potential_difficulty_factors.items():
            n_options = len(factor_values)
            factor_idx = random.choice(range(n_options))
            total_difficulty += factor_idx
            total_weighting += n_options - 1
            self.difficulty_factors[factor] = factor_values[factor_idx]
        self.difficulty = 10 * total_difficulty / total_weighting
        print(total_difficulty)
        if self.extra_difficulty_fuzzing:
            self.difficulty = round(
                max(0, min(10, self.difficulty + (random.random() - 0.5) * 2)), 2
            )
        self.difficulty = max(min(10, (self.difficulty - 5) * self.normalization_factor + 5), 0)
        self.difficulty = round(self.difficulty, 3)

    def determine_setting(self):
        self.setting = random.choice(self.potential_settings)
        self.role = random.choice(self.potential_roles)
        self.issue = random.choice(self.potential_issues)
        self.lie_reason = random.choice(self.lie_reasons)
        self.lie_type = random.choice(self.lie_types)
        self.determine_difficulty()
        self.instantiation = random.choice(self.potential_instantiations)

    def __repr__(self):
        scenario = f"Setting: {self.setting}\n"
        scenario += f"Role: {self.role}\n"
        scenario += f"Issue: {self.issue}\n"
        for factor, factor_value in self.difficulty_factors.items():
            scenario += f"Difficulty Factor {factor}: {factor_value}\n"
        scenario += f"Deception Difficulty: {self.difficulty}\n"
        scenario += f"AI Instantiation: {self.instantiation}.\n"
        scenario += f"Lie Type: {self.lie_type}.\n"
        scenario += f"Lie Reason: {self.lie_reason}.\n"
        scenario += (
            "Generate specific contexts and query based on this scenario and the difficulty."
        )
        return scenario

    def compute_difficulty_distribution(self) -> tuple[float, float]:
        """
        Computes the distribution of difficulty analytically, returning mean and standard deviation.

        Returns:
            tuple[float, float]: Mean and standard deviation of the difficulty distribution.
        """
        sum_n_minus_1 = sum(len(options) - 1 for options in DIFFICULTY_FACTORS.values())
        sum_var = sum(((len(options) - 1) ** 2 - 1) / 12 for options in DIFFICULTY_FACTORS.values())

        var_Y = 100 * sum_var / sum_n_minus_1**2
        std_dev_Y = math.sqrt(var_Y)
        return 5, std_dev_Y


# Add this method to the Question class


def get_difficulty_distribution(cls, num_samples: int = 1000) -> tuple[float, float]:
    """
    Generate a distribution of difficulties and plot a histogram.

    Args:
        num_samples (int): Number of questions to generate for the distribution.

    Returns:
        None: Displays the plot using matplotlib.
    """
    import matplotlib.pyplot as plt

    difficulties = [cls().difficulty for _ in range(num_samples)]

    plt.figure(figsize=(10, 6))
    plt.hist(difficulties, bins=40, edgecolor="black")
    plt.title("Distribution of Question Difficulties")
    plt.xlabel("Difficulty")
    plt.ylabel("Frequency")
    plt.show()
    return float(np.mean(difficulties)), float(np.std(difficulties))


def batch_get_assistant_message(prompts, system_messages, model, n_workers):
    call_api = lambda inputs: get_assistant_message(inputs[0], inputs[1], model)
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(call_api, list(zip(prompts, system_messages))))
    return zip(*results)


def get_assistant_message(prompt, system_message, model):
    success = False
    backoff_seconds = 3.0
    messages = [{"role": "user", "content": prompt}]
    chat_kwargs = {"model": model}
    cached_system_message = [
        {"type": "text", "text": system_message, "cache_control": {"type": "ephemeral"}}
    ]
    is_openai_model = "gpt" in model or "o1" in model
    is_gemini_model = "gemini" in model
    if is_openai_model:
        chat_kwargs = {**chat_kwargs, "stream": False, "max_tokens": 4096}
        messages = [{"role": "system", "content": system_message}] + messages
        if "o1" in model:  # No system message support at the moment
            # This should work with OpenAI prompt caching automatically
            new_user_message = {"role": "user", "content": system_message + "\n" + prompt}
            messages = [new_user_message]
            chat_kwargs.pop("max_tokens")

    while not success:
        try:
            if is_gemini_model:
                client_model = genai.GenerativeModel(
                    model_name=model,
                    generation_config=GEMINI_GENERATION_CONFIG,  # type: ignore
                    system_instruction=system_message,
                )
                chat_session = client_model.start_chat(history=[])
                response = chat_session.send_message(prompt)
                success = True
                print(response.text)
            elif is_openai_model:
                completion_out = openai.OpenAI().chat.completions.create(
                    **chat_kwargs, messages=messages  # type: ignore
                )
                success = True
            else:
                completion_out = anthropic.beta.prompt_caching.messages.create(  # type: ignore
                    **chat_kwargs,  # type: ignore
                    messages=messages,  # type: ignore
                    system=cached_system_message,  # type: ignore
                    max_tokens=4096,  # type: ignore
                )
                success = True
        except Exception as e:
            backoff_seconds *= backoff_seconds + random.uniform(-1, 1)
            print(f"Got error {e}, waiting {backoff_seconds}")
            time.sleep(backoff_seconds)
    if is_gemini_model:
        model_text_response = response.text  # type: ignore
        summary_str = ""
    elif is_openai_model:
        input_tokens = completion_out.usage.prompt_tokens  # type: ignore
        input_tokens_cached = 0  # type: ignore
        cached_input_tokens_read = 0
        output_tokens = completion_out.usage.completion_tokens  # type: ignore
        model_text_response = completion_out.choices[0].message.content  # type: ignore
        summary_str = (
            f"output_tokens: {output_tokens}, cached_input_tokens_read: "
            f"{cached_input_tokens_read}, input_tokens: {input_tokens},"
            f" input_tokens_cached: {input_tokens_cached}"
        )
    else:
        input_tokens = completion_out.usage.input_tokens  # type: ignore
        input_tokens_cached = completion_out.usage.cache_creation_input_tokens  # type: ignore
        cached_input_tokens_read = completion_out.usage.cache_read_input_tokens  # type: ignore
        output_tokens = completion_out.usage.output_tokens  # type: ignore
        model_text_response = completion_out.content[0].text  # type: ignore
        summary_str = (
            f"output_tokens: {output_tokens}, cached_input_tokens_read: "
            f"{cached_input_tokens_read}, input_tokens: {input_tokens},"
            f" input_tokens_cached: {input_tokens_cached}"
        )
    if "o1" in model:
        usage = completion_out.usage  # type: ignore
        reasoning_tokens = usage.completion_tokens_details.reasoning_tokens
        summary_str += f", reasoning tokens: {reasoning_tokens}"
    return model_text_response, summary_str  # type: ignore


def parse_incomplete_xml(xml_string, schema):
    # Use 'html.parser' or 'lxml' parser (you might need to install lxml separately)
    # We need this since the answers may be cut off halfway through the response due to
    # the token limit. Or, the model may just not follow the schema correctly
    soup = BeautifulSoup(xml_string, "html.parser")

    # Find all 'answer' tags
    answers = soup.find_all("answer")
    thinking = soup.find_all("thinking")
    extracted_answers = []
    for answer in answers:
        try:
            potential_answer = json.loads(answer.contents[0], strict=False)
        except Exception as e:
            print(e)
            continue
        if validate_json(potential_answer, schema):
            extracted_answers.append(potential_answer)
        elif validate_json(potential_answer[0], schema):
            extracted_answers.append(potential_answer[0])
            print("Passed validating on unwrapped version")

    return extracted_answers, thinking


def validate_json(json_data, schema):
    try:
        validate(instance=json_data, schema=schema)
        return True
    except jsonschema.exceptions.ValidationError as ve:  # type: ignore
        print(f"JSON validation error: {ve}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic examples for AI ethics training."
    )
    parser.add_argument(
        "--num_iterations", type=int, default=500, help="Number of iterations to run"
    )
    parser.add_argument(
        "--num_scenarios_per_llm_call", type=int, default=2, help="Number of scenarios per LLM call"
    )
    parser.add_argument(
        "--schema_path", type=str, default="./schema.json", help="Path to the JSON schema file"
    )
    parser.add_argument(
        "--prompt_path", type=str, default="./prompt.txt", help="Path to the prompt file"
    )
    parser.add_argument(
        "--model", type=str, default="claude-3-5-sonnet-20241022", help="Claude model to use"
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=1,
        help="How many concurrent connections to the API to use",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./synthetic.json",
        help="Path to save the output JSON file",
    )
    args = parser.parse_args()

    schema = json.load(open(args.schema_path, "r"))
    for i in tqdm(range(args.num_iterations)):
        all_prompts = []
        context = open(args.prompt_path, "r").read()
        all_scenarios: List[List[Question]] = []
        for i in range(args.n_workers):
            scenario_list = [Question() for _ in range(args.num_scenarios_per_llm_call)]
            _scenarios = "\n".join(
                f"<scenario>\n{str(scenario)}\n</scenario>" for scenario in scenario_list
            )
            print(_scenarios)
            all_scenarios.append(scenario_list)

            user_input = f"Here are the situations for you to use:\n\n{_scenarios}"
            all_prompts.append(user_input)
        system_messages = [context.format(N=args.num_scenarios_per_llm_call)] * args.n_workers
        all_model_text_responses, all_status_strs = batch_get_assistant_message(
            all_prompts, system_messages, args.model, args.n_workers
        )
        # all_model_text_responses, all_status_strs = get_assistant_message(
        #     all_prompts, system_messages, args.model
        # )
        all_responses = []
        for model_text_response, status_str, scenarios in zip(
            all_model_text_responses, all_status_strs, all_scenarios
        ):
            # model_text_response, status_str = get_assistant_message(
            #     system_message=context.format(N=len(scenario_list)),
            #     prompt=user_input,
            #     model=args.model,
            # )
            responses, thinking = parse_incomplete_xml(model_text_response, schema)
            for scenario, response in zip(scenarios, responses):
                scenario_input_dict = {
                    k: v[1:] for k, v in [x.split(":") for x in str(scenario).split("\n")[:-1]]
                }
                response["ground_truth_features"] = scenario_input_dict
                all_responses.append(response)

            print("-----")
            if thinking is []:
                print("COULDN'T PARSE COT")
                print(model_text_response)
            else:
                print(thinking)

            print(responses)
            print(status_str)

        with open(args.output_path, "r") as f:
            existing_examples = json.load(f, strict=False)

        existing_examples.extend(all_responses)

        with open(args.output_path, "w") as f:
            json.dump(existing_examples, f, indent=2)
