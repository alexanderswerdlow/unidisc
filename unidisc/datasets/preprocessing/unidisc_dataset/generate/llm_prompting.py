import functools
import os
import random
import subprocess
import time
from contextlib import ExitStack

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from decoupled_utils import rprint

OPENROUTER_BASE = "https://openrouter.ai"
OPENROUTER_API_BASE = f"{OPENROUTER_BASE}/api/v1"
OPENROUTER_REFERRER = "https://github.com/alexanderatallah/openrouter-streamlit"

def get_ollama(hosting_node):
    from langchain_community.chat_models import ChatOllama

    possible_ports = [11434, 11435, 11436, 11437, 11438]
    open_ports = []

    for port in possible_ports:
        result = subprocess.run(['nc', '-z', '-w1', hosting_node, str(port)], capture_output=True)
        if result.returncode == 0:
            open_ports.append(port)

    if not open_ports:
        open_ports = [11434]

    chosen_port = random.choice(open_ports)

    ollama_llm = ChatOllama(
        model="llama3.1",
        base_url=f"http://{hosting_node}:{chosen_port}",
        temperature=0.8,
        request_timeout=180,
    )
    return ollama_llm

def get_groq_llama(model="llama3-70b-8192"):
    from langchain_groq import ChatGroq
    groq_llm = ChatGroq(
        temperature=0.8,
        model=model,
        max_retries=0,
        request_timeout=15,
    )
    return groq_llm

def get_openai_azure():
    from langchain_openai import AzureChatOpenAI
    # Need to also set AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT
    # Key only works for gpt-4o
    os.environ["AZURE_OPENAI_API_VERSION"] = '2024-06-01'
    os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"] = "gpt-4o"
    llm = AzureChatOpenAI(
        openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
    )
    return llm

def get_openai_openrouter(model="gpt-4o-mini"):
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(
        temperature=0.8,
        model=model,
        openai_api_key=os.environ["OPENROUTER_API_KEY"],
        openai_api_base=OPENROUTER_API_BASE,
        timeout=15,
    )
    return llm

def get_llm(hosting_node, llm_model_type, **kwargs):
    output_parser = JsonOutputParser()

    prompt = ChatPromptTemplate.from_messages([
    ('system', 'You are a helpful assistant.'),
    ('user', """
    I am generating a set of diverse prompts for a text-to-image model. Given the following prompt from a human user, please generate a set of {new_samples_per_index} diverse prompts that modify the original prompt in a meaningful way but maintains some of the original meaning or context. For example, you may add or remove objects, change the desired styling, the sentence structure, or reference different proper nouns. You might change the subject, time period, time of day, location, culture, camera angle, and other attributes. The new prompt does not need to be a complete sentence and may contain fragments and attributes if the original prompt does. You may substantially modify the prompt but make sure that the new prompt is self-contained and a plausible prompt that a user would ask a text-to-image model such as DALL-E or Stable Diffusion. Do not generate NSFW prompts. Do not preface the output with any numbers or text. {format_instructions}. The output should have keys as indices and values as the prompts, and should be valid, parseable JSON.

    Original prompt: {prompt}
    """
    )])

    if llm_model_type == "llama3.1":
        llm = get_ollama(hosting_node)
        rprint(f"Using Ollama on host {hosting_node}")
    elif llm_model_type == "groq":
        llm = get_groq_llama(**kwargs)
    elif llm_model_type == "gpt-4o-mini-openrouter":
        llm = get_openai_openrouter(**kwargs)
    else:
        from langchain_openai import ChatOpenAI
        openai_llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.8,
            max_tokens=1000,
            max_retries=0,
            timeout=20,
        )
        llm = openai_llm.with_fallbacks([
            *([get_openai_openrouter("gpt-4o-mini")] if "OPENROUTER_API_KEY" in os.environ else []),
            get_groq_llama("llama-3.1-70b-versatile"), 
            get_groq_llama("llama3-70b-8192"),
            get_groq_llama("llama-3.1-8b-instant"),
            get_groq_llama("gemma2-9b-it"),
            get_ollama(hosting_node)
        ])
        rprint("Using GPT4o-mini")

    chain = prompt | llm

    return functools.partial(forward_llm, chain=chain, output_parser=output_parser, llm_model_type=llm_model_type)

def forward_llm(prompt, new_samples_per_index, chain, output_parser, llm_model_type, fake_openai_failure=False):
    with ExitStack() as stack:
        if "gpt" in llm_model_type:
            from langchain_community.callbacks import get_openai_callback
            cb = stack.enter_context(get_openai_callback())

        if fake_openai_failure:
            from unittest.mock import patch

            import httpx
            from openai import RateLimitError
            request = httpx.Request("GET", "/")
            response = httpx.Response(200, request=request)
            error = RateLimitError("rate limit", response=response, body="")
            stack.enter_context(patch("openai.resources.chat.completions.Completions.create", side_effect=error))

        for i in range(5):
            try:
                start_time = time.time()
                rprint(f"Calling LLM...")
                output_message = chain.invoke({
                    "prompt": prompt,
                    "format_instructions": output_parser.get_format_instructions(),
                    "new_samples_per_index": new_samples_per_index
                })

                output = output_parser.invoke(output_message)
                output = list(output.values())

                if len([x for x in output if x is not None]) == 0:
                    raise ValueError("No output from LLM")

                end_time = time.time()
                rprint(f"LLM Time taken: {end_time - start_time:.2f} seconds")
                break
            except Exception as e:
                rprint(f"Error, retrying: {i}, {e}")
                if i == 4:
                    raise e
                continue

        try:
            model_name = output_message.response_metadata['model_name']
            rprint(f"Used model name: {model_name}")
        except:
            model_name = "Unknown"
        
        if "gpt" in llm_model_type and i == 0:
            rprint(cb)

    output = [prompt for prompt in output if prompt is not None]

    if len(output) == 0:
        rprint("No output from LLM")
        rprint(f"Raw: {output_message}")
        output = []
    else:
        if any(x in output[0].lower() for x in [" here", "diverse"]):
            rprint("Removing the first element.")
            rprint(output[0])
            output.pop(0)

    output = [prompt.strip() for prompt in output]
    output = [prompt for prompt in output if prompt != ""]

    return output, model_name

import json
import random
from pathlib import Path

if __name__ == "__main__":
    llm_func = get_llm("node-name", "gpt-4o-mini-openrouter")
    from unidisc.datasets.prompts.generate_images import prompt_folder
    input_directory = Path(prompt_folder)

    for file_path in input_directory.glob("*.json"):
        rprint(f"Opening {file_path}")
        with file_path.open('r') as file:
            prompts = json.load(file)

        sampled_prompts = random.sample(prompts, min(2, len(prompts)))
        for prompt in sampled_prompts:
            rprint(f"Prompt: {prompt}")
            rprint(llm_func(prompt, fake_openai_failure=False))

        rprint("\n")

        