from DataGenerator import DataGenerator
import time

generator = DataGenerator('datasets/bas.txt', batch_size=256, history_size=500, device='cpu')

print("getting batch")
start = time.time()
batch = generator.get_batch()
print("got batch")
end = time.time()
print("time taken: {} seconds".format(end-start))
