import argparse

def parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Bert pretrained config", type=str, default="bert-base", choices=["bert-base", "bert-large"])
    parser.add_argument("--lora", help="lora finetune", action="store_true")
    parser.add_argument("--full_finetune", help="allow pretrained encoder trainable", action="store_true")
    
    parser.add_argument("--outdir", default='outdir')
    
    args = parser.parse_args()
    if args.config=="bert-base":
        if args.lora: from .config import ConfigBertBaseLora as Config
        else: from .config import ConfigBertBase as Config
    elif args.config=="bert-large":
        if args.lora: from .config import ConfigBertLargeLora as Config
        else: from .config import ConfigBertLarge as Config
    else:
        assert (0), "Invalid config"
    
    return args, Config