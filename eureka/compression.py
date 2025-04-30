import pickle

def return_pickled_object(content):
    """
    Pickle the given content and return it.
    """
    return pickle.dumps(content)

def return_unpickled_object(content):
    """
    Unpickle the given content and return it.
    """
    return pickle.loads(content)

def read_file_and_pickle(file_path):
    """
    Read the content of a file and pickle it.
    """
    with open(file_path, 'rb') as file:
        content = file.read()
    return pickle.dumps(content)

def read_rollout_and_pickle(file_path):
    """
    Read the content of a rollout file and pickle it.
    """
    with open(file_path, 'rb') as file:
        content = file.read()
    score = float(content.split(b'\n')[0].split(b' ')[-1])
    observations = content.split(b'\n')[1:]
    for i in range(len(observations)):
        if observations[i].decode('utf-8').strip():
            observations[i] = eval(observations[i].decode('utf-8').strip())
    # print(score)
    # print(observations[0:2])
    return pickle.dumps({'score': score, 'observations': observations})

if __name__ == "__main__":
    # Example usage
    file_path = "/home/avidavid/Eureka/eureka/preference_data/ShadowHand_2025-03-07_01-39-39.txt"
    pickled_content = read_rollout_and_pickle(file_path)
    # Write pickled content to a file
    with open("pickled_content.pkl", "wb") as f:
        f.write(pickled_content)
    # print("Pickled content:", pickled_content)