import sys
import functions as f
import vars as v

# ================================================================

History = ""

if not f.setdevice():
    sys.exit("Could not find CUDA")

Model, Tokenizer = f.setmodel(v.ModelName, v.QuantConfig)
SummModel, SummTokenizer = f.setmodel(v.SummModelName, v.QuantConfig)

f.line()

while True:
    UserPrompt = input("User: ")
    f.line()

    if UserPrompt == "!break":
        sys.exit()

    HisSystemPrompt = v.SystemPrompt + f" Following is the chat history. {History}"

    Inputs = f.prompt(Model, Tokenizer, UserPrompt, HisSystemPrompt)
    Outputs = f.response(Model, Tokenizer, Inputs)

    f.line()
    print(f"{v.Who}: {Outputs}")
    f.line()

    History = f.summ(SummModel, SummTokenizer, UserPrompt, Outputs, v.SummSystemPrompt)

