import os
import sys
import collections


VOCAB_SIZE = 500
DATA_DIR = 'data'


def convert_to_ids(SOURCE):

    with open(SOURCE, 'r') as f:
        chords = f.read().split()

    init_count = collections.Counter(chords).most_common()
    unk_count = sum([n for _, n in init_count[VOCAB_SIZE-1:]])
    count = [('UNK', unk_count)] + init_count[:VOCAB_SIZE-1]

    os.makedirs(DATA_DIR, exist_ok=True)

    vocabulary_path = os.path.join(DATA_DIR, 'vocabulary')
    with open(vocabulary_path, 'w') as f:
        f.write('{}\n'.format(VOCAB_SIZE))
        f.write('\n'.join(['{} {}'.format(word, n) for word, n in count]))

    dictionary = dict()
    for i, (word, _) in enumerate(count):
        dictionary[word] = i

    data_as_ids = []
    for chord in chords:
        if chord in dictionary:
            data_as_ids.append(str(dictionary[chord]))
        else:
            data_as_ids.append('0')

    ids_path = os.path.join(DATA_DIR, 'data_as_ids')
    with open(ids_path, 'w') as f:
        f.write(' '.join(data_as_ids))

if __name__ == "__main__":
    convert_to_ids(sys.argv[1])
