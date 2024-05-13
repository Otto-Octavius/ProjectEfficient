from llama.tokenizer import Tokenizer
from llama.model import ModelArgs, Llama
import torch
import time


def inference():
    torch.manual_seed(1)

    tokenizer_path = "/project/saifhash_1190/llama2-7b/tokenizer.model"
    model_path = "/home1/devavara/EE599-Project/ml-systems-final-project-SamDJ101-main/Amp_GA.pth"

    tokenizer = Tokenizer(tokenizer_path)

    checkpoint = torch.load(model_path, map_location="cpu")
    model_args = ModelArgs()
    torch.set_default_tensor_type(torch.cuda.HalfTensor) # load model in fp16
    model = Llama(model_args)
    model.load_state_dict(checkpoint, strict=False)
    model.to("cuda")
    
    prompts = [
        # For these prompts, the expected answer is the natural continuation of the prompt
     "Best Hiking place in Los Angeles for sunset is",
    """A brief message congratulating the team on the launch:

        Hi everyone,
        
        I just """
    
    ]

    model.eval()
    results = model.generate(tokenizer, prompts, max_gen_len=64, temperature=0.6, top_p=0.9)

    for prompt, result in zip(prompts, results):
        print(prompt)
        print(f"> {result['generation']}")
        print("\n==================================\n")

    
if __name__ == "__main__":
    s = time.time()
    inference()
    e = time.time()
    print(e-s)