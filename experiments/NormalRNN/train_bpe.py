from lib.ByteLevelBPE import ByteLevelBPE

text = open("../../datasets/bas_clean.txt").read()[-1000000:]

bpe = ByteLevelBPE.learn(text, vocab_size=2048)

bpe.save("../../datasets/bpe_2048.json")
