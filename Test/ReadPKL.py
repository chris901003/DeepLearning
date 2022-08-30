import pickle


file_path = '/Users/huanghongyan/Documents/DeepLearning/pytorch_geometric/TemporalGestureRecognition/PoseVideo/' \
            'extract.pkl'

file = open(file_path, 'rb')
content = list()
while True:
    try:
        C = pickle.load(file)
        content.append(C)
    except EOFError:
        break
print(content)
