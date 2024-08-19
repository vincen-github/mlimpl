import string
from re import compile
from torch import triu, ones
from DeepLearning.Transformer.params import pad_id, bos_id, eos_id, device


def is_contain_zh(word):
    pattern = compile(r'[\u4e00-\u9fa5]')
    match = pattern.search(word)
    return True if match else False


def get_pad_tensor(encoder_input, pad_id, decoder_input=None):
    # pad_matrix is used to indicate whether the word is <pad> or not. we will use it to decide which attention scores will be masked.
    pad_tensor = (encoder_input == pad_id)
    if decoder_input is None:
        # Now we have tensor of shape [batch_size, seq_len]. we need to transform it to [batch_size, seq_len, seq_len] and then apply it to masked_fill method.
        # Here we utilize the broadcasting mechanism.
        pad_tensor = pad_tensor.unsqueeze(-1) | pad_tensor.unsqueeze(-2)
    else:
        # otherwise we will return encoder_decoder_pad_tensor
        # note that encoder_input tell us where is <pad> in encoder_output. It will be used to build encoder_decoder_pad_tensor.
        pad_tensor = (decoder_input == pad_id).unsqueeze(-1) | pad_tensor.unsqueeze(-2)
        # Note that encoder_decoder_output_pad_tensor is not a square matrix as the number of words of english sentence is different with corresponding chinese sentence.
    return pad_tensor


def get_masked_tensor(decoder_input, pad_id):
    # pad_tensor is used to indicate whether the word is <pad> or not.
    decoder_input_pad_tensor = (decoder_input == pad_id)
    decoder_input_pad_tensor = decoder_input_pad_tensor.unsqueeze(-1) | decoder_input_pad_tensor.unsqueeze(-2)
    # decoder_input_masked_tensor is aim to prevent the latter words of current word from being seen by decoder.
    decoder_input_masked_tensor = triu(
        ones(decoder_input.shape[0], decoder_input.shape[1], decoder_input.shape[1], dtype=bool, device=device),
        diagonal=1) | decoder_input_pad_tensor
    return decoder_input_masked_tensor


def indices2sentence(indices, vocab, is_english):
    indices = [index for index in indices if index not in (pad_id, bos_id, eos_id)]
    if is_english:
        english = ' '.join(vocab.lookup_tokens(indices))
        for punctuation in string.punctuation:
            english = english.replace(' ' + punctuation, punctuation)
        return english
    return ''.join(vocab.lookup_tokens(indices))
