import random, string

charset_base = string.ascii_lowercase + string.ascii_uppercase

def get_nonesense_label(char_weights):
    word_len = random.randint(0, 10)

    if word_len == 0:
        return ''

    nonesense_label = random.choices(charset_base, weights=char_weights, k=word_len)
    return ''.join(nonesense_label)

def random_insertion(text, char_weights):
    idx = random.randint(0, len(text))
    char_in = random.choices(charset_base, weights=char_weights, k=1)
    return text[:idx] + char_in[0] + text[idx:]

def random_deletion(text, char_weights):
    idx = random.randint(0, len(text) - 1)
    return text[:idx] + text[idx + 1:]

def random_swap(text, char_weights):
    text = list(text)
    indices = random.choices([idx for idx in range(len(text))], k=2)
    temp = text[indices[0]]
    text[indices[0]] = text[indices[1]]
    text[indices[1]] = temp
    return ''.join(text)

def random_replacement(text, char_weights):
    idx = random.randint(0, len(text) - 1)
    new_char = random.choices(charset_base, weights=char_weights, k=1)
    text = list(text)
    text[idx] = new_char[0]
    return ''.join(text)

def random_edits(text, char_weights):
    edit_funcs = [random_insertion, random_deletion, random_swap, random_replacement]
    edits = random.choices(edit_funcs, k=3)
    text = edits[0](text, char_weights)
    text = edits[1](text, char_weights)
    return text

def random_multi_insert(text, char_weights):
    if len(text) > 2:
        num_pos = random.randint(1, 2)
    else:
        num_pos = 1
    for i in range(num_pos):
        if len(text) == 0:
            text = random.choices(charset_base, weights=char_weights, k=1)[0]
        idx = random.randint(0, len(text))
        char_in = text[-1] if idx == len(text) else text[idx]
        seq_in = char_in * random.randint(2, 5)
        text = text[:idx] + seq_in + text[idx:]
    return text[:25] if len(text) > 25 else text