import functools
import os
import random
import subprocess
import time
from contextlib import ExitStack
from decoupled_utils import rprint

OPENROUTER_BASE = "https://openrouter.ai"
OPENROUTER_API_BASE = f"{OPENROUTER_BASE}/api/v1"
OPENROUTER_REFERRER = "https://github.com/alexanderatallah/openrouter-streamlit"

def get_groq_llama(model="llama-3.2-90b-text-preview"):
    from langchain_groq import ChatGroq
    groq_llm = ChatGroq(
        temperature=0.8,
        model=model,
        max_retries=0,
        request_timeout=30,
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

def get_llm(llm_model_type, **kwargs):
    from langchain_core.output_parsers import JsonOutputParser
    output_parser = JsonOutputParser()

    from langchain_core.prompts import ChatPromptTemplate
    prompt = ChatPromptTemplate.from_messages([
    ('system', 'You are a helpful assistant.'),
    ('user', """
    I am generating a set of incorrect captions for an image. Given the following prompt from a human user that corresponds to a real image, please generate a set of 12 incorrect prompts that modify the original prompt but maintains some of the original meaning or context. For example, you may add or remove an object, change the desired styling, the sentence structure, or reference a different proper noun. You might change the subject, time period, time of day, location, culture, camera angle, and other attributes. Make the prompts very simple and do not use very exotic or rare objects or words. For half of the captions, make them broken, have improper grammar or just be nonsensical. Do not generate NSFW prompts. Do not preface the output with any numbers or text. {format_instructions}. The output should have keys as indices and values as the prompts, and should be valid, parseable JSON. Make sure to escape quotes.

    Original prompt: {prompt}
    """)
    ])


    openai_llm = get_groq_llama("llama-3.2-90b-text-preview")
    llm = openai_llm.with_fallbacks([
        get_groq_llama("llama-3.2-11b-text-preview"),
        get_groq_llama("gemma-7b-it"),
        get_groq_llama("llama-3.2-3b-preview"),
        get_groq_llama("llama-3.2-1b-preview"),
        get_groq_llama("llama-3.2-11b-vision-preview"),
        get_groq_llama("llama-3.2-90b-vision-preview"),
    ])

    chain = prompt | llm

    return functools.partial(forward_llm, chain=chain, output_parser=output_parser, llm_model_type=llm_model_type)

def forward_llm(prompt, chain, output_parser, llm_model_type, fake_openai_failure=False):
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

        for i in range(10):
            try:
                start_time = time.time()
                rprint(f"Calling LLM...")
                output_message = chain.invoke({
                    "prompt": prompt,
                    "format_instructions": output_parser.get_format_instructions()
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
                if i == 9:
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
    llm_func = get_llm(llm_model_type="")
    res = llm_func("A red sailboat on a blue ocean with a yellow sun", fake_openai_failure=False)
    breakpoint()

        