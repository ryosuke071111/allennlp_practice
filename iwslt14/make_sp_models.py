import sentencepiece as spm

DATA_DIR = "/home/ryosuke/desktop/allen_practice/iwslt14/iwslt14.tokenized.de-en/tmp/"

spm.SentencePieceTrainer.Train(
    f'--input={DATA_DIR}train.en --model_prefix=spm_en --character_coverage=0.9995 --vocab_size=8000 --pad_id=0 \
    --unk_piece=@@UNKNOWN@@ --pad_piece=@@PADDING@@  --bos_piece=@start@ --eos_piece=@end@ --unk_id=1 --bos_id=2 --eos_id=3 --hard_vocab_limit=false'
)
spm.SentencePieceTrainer.Train(
    f'--input={DATA_DIR}train.de --model_prefix=spm_de --character_coverage=0.9995 --vocab_size=8000 --pad_id=0 \
     --unk_piece=@@UNKNOWN@@ --pad_piece=@@PADDING@@  --bos_piece=@start@ --eos_piece=@end@  --unk_id=1 --bos_id=2 --eos_id=3 --hard_vocab_limit=false'
)