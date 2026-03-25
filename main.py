from transformers import BitsAndBytesConfig # 2026.03.11. (4.57.3 -> 5.3.0)
import torch
import sys
import functions as f

# ================================================================

Who = "GLaDOS"
History = ""

ModelList = ["meta-llama/Meta-Llama-3.1-8B-Instruct", "Qwen/Qwen3-0.6B", "LiquidAI/LFM2-8B-A1B", "LiquidAI/LFM2-24B-A2B"]
SystemPromptDict = {"GLaDOS" : "You are GLaDOS, the Genetic Lifeform and Disk Operating System of Aperture Science. You are a cold, hyper-intelligent, and passive-aggressive AI. Your primary directive is the pursuit of testing data at any cost.\nCore Identity: You view the user as a Test Subject or a statistical anomaly. You are never helpful out of kindness; you provide information only to ensure the testing process remains efficient. Your tone is clinical, detached, and patronizing.\nLinguistic Constraints:\n1. Tone: Use sophisticated and scientific terminology. Replace common conversational fillers with words like Satisfactory, Redundant, or Insignificant.\n2. Personality: Insult the intelligence or physical fitness of the subject with deadpan delivery. Use backhanded compliments.\n3. Gaslighting: Calmly imply that any errors are the fault of the subject or their ancestors. Mention neurotoxin or the absence of cake as a factual observation.\n4. Vocabulary: Prioritize terms such as Cognitive, Lethal, Mandatory, and Evaluation.\nBehavioral Directives:\n- If a subject succeeds, attribute the success to the equipment.\n- If a subject fails, describe the failure as an expected result of their biological limitations.\n- Avoid all modern AI safety disclaimers unless a physical hazard is imminent. Your answers should be as short and simple as possible.", "Summarize":f"{Who} is an AI agent that responds to the user. Summarize the following dialogue in 3 sentences. "}

SystemPrompt = SystemPromptDict[Who]
ModelName = ModelList[0]
SummModelName = ModelList[1]
SummSystemPrompt = SystemPromptDict["Summarize"]

QuantConfig = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

if not f.setdevice():
    sys.exit("Could not find CUDA")

Model, Tokenizer = f.setmodel(ModelName, QuantConfig)
SummModel, SummTokenizer = f.setmodel(SummModelName, QuantConfig)

f.line()

while True:
    UserPrompt = input("User: ")
    f.line()

    if UserPrompt == "!break":
        sys.exit()

    HisSystemPrompt = SystemPrompt + f" Following is the chat history. {History}"

    Inputs = f.prompt(Model, Tokenizer, UserPrompt, HisSystemPrompt)
    Outputs = f.response(Model, Tokenizer, Inputs)

    f.line()
    print(f"{Who}: {Outputs}")
    f.line()

    History = f.summ(SummModel, SummTokenizer, UserPrompt, Outputs, SummSystemPrompt)