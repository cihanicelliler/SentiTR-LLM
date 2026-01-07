from models.encoder import SentimentEncoder
from models.llm import SentimentLLMClassifier
from training.trainer import SentiTRTrainer
from utils.logger import setup_logger

logger = setup_logger("FineTuner")

def train_encoder_model(args, dataset):
    """
    Fine-tunes an Encoder-based model (BERT/ELECTRA).
    """
    logger.info(f"Initializing Encoder model: {args.model_name}")
    model = SentimentEncoder(model_name=args.model_name, num_labels=3)
    
    def tokenize_function(examples):
        return model.tokenizer(examples["text"], truncation=True, padding=False, max_length=512)
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    # Trainer with remove_unused_columns=True (default) will handle removing 'text'
    # But we set it to False in trainer.py. We should fix trainer.py or manually remove here.
    # Let's assume we fix trainer.py back to default or True.
    
    trainer = SentiTRTrainer(
        model=model.model,
        tokenizer=model.tokenizer,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['validation'],
        output_dir=f"checkpoints/{args.model_name.split('/')[-1]}",
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate
    )
    
    metrics = trainer.train()
    trainer.evaluate()

    return trainer, tokenized_dataset

def train_llm_model(args, dataset):
    """
    Fine-tunes an LLM using QLoRA.
    """
    logger.info(f"Initializing LLM model: {args.model_name}")
    # Use the classifier wrapper for direct sequence classification training
    model_wrapper = SentimentLLMClassifier(model_name=args.model_name, num_labels=3, use_4bit=args.use_4bit)
    
    # LLMs require max_length padding/truncation during tokenization usually handled in collation
    # But HF Trainer handles basic padding. We ensured tokenizer pad token is set in wrapper.
    
    # Tokenize for LLM
    def tokenize_function(examples):
        return model_wrapper.tokenizer(examples["text"], truncation=True, padding=False, max_length=512)
        
    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    trainer = SentiTRTrainer(
        model=model_wrapper.model,
        tokenizer=model_wrapper.tokenizer,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['validation'],
        output_dir=f"checkpoints/{args.model_name.split('/')[-1]}",
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        fp16=True # Likely needed for LLM
    )
    
    metrics = trainer.train()
    trainer.evaluate()

    return trainer, tokenized_dataset
