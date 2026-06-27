import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import re
import sys
import torch
import vars as v
from model import HistoryLM

def line():
    print("\n", "="*64, "\n")

def setdevice():
    if torch.cuda.is_available():
        return True
    else:
        return False

def template(model, input):
    if model.has_user_template:
        input = model.user_template.format(chat_input=input)
    return model.few_shot_text + input

def main():
    
    if not setdevice():
        sys.exit("Could not find CUDA")
        
    MainModel = HistoryLM(v.MainInfoDir)
    SummModel = HistoryLM(v.SummInfoDir)

    line()

    while True:
        UserPrompt = input("User: ")
        if UserPrompt == "!break":
            sys.exit()
        TempUserPrompt = template(MainModel, UserPrompt)
        MainModel.messages.append({"role": "user", "content": TempUserPrompt})
        line()

        # Main Model
        Inputs = MainModel.tokenize()
        print(f"Model: ", end = "")
        Outputs = MainModel.response(Inputs)
        
        line()
        
        # Summarize Model
        History = []
        for context in [UserPrompt, Outputs]:
            if len(context) >= 128:
                HisContext = template(SummModel, context)
                SummModel.messages.append({"role": "user", "content": HisContext})
                HisInputs = SummModel.tokenize()
                History.append(re.sub(r"['\"\s]",r"",SummModel.response(HisInputs)))
                SummModel.messages.pop(-1)
            else:
                History.append(context)
            
        
        MainModel.messages.pop(-1)
        MainModel.messages.append({"role": "user", "content": History[0]})
        MainModel.messages.append({"role": "assistant", "content": History[1]})
        
if __name__ == "__main__":
    main()
    print(re.sub(r"['\"]",r"","test"))