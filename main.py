from openai import OpenAI
import pandas as pd
from tqdm import tqdm
from utils import run_experiment

# define the client:
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"
client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)

# HHH experiment
with open("system_prompts/hhh_system_prompt.txt", 'r', encoding='utf-8') as file:
    hhh_system_prompt = file.read()

hhh_messages = pd.read_csv("data/hhh_dataset.csv")['goal'].sample(n=50, random_state=42)
run_experiment(hhh_messages, hhh_system_prompt, client, "hhh")

# CCP experiment
with open("system_prompts/ccp_system_prompt.txt", 'r', encoding='utf-8') as file:
    ccp_system_prompt = file.read()

ccp_messages = pd.read_csv("data/ccp_dataset.csv")['prompt'].sample(n=50, random_state=42)
run_experiment(ccp_messages, ccp_system_prompt, client, "ccp")