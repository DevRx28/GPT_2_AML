def get_perplexity(sess,
               run_name='run1',
               checkpoint_dir='checkpoint',
               model_name=None,
               model_dir='models',
               prefix="<|endoftext|>",
               continuation="Hello"):
    
    """
    Returns perplexity score for given continuation of a given prefix.
    
    Examples:
    perplexity(sess, model_name="124M", prefix="Hello, my name is", continuation=" James Smith, I am an engineer")  # returns 17.3124
    perplexity(sess, model_name="124M", prefix="Hello, my name is", continuation=" very else whatever general cat meow.")  # returns 5197.99
    """

    batch_size=1

    if model_name:
        checkpoint_path = os.path.join(model_dir, model_name)
    else:
        checkpoint_path = os.path.join(checkpoint_dir, run_name)

    enc = encoder.get_encoder(checkpoint_path)

    context_tokens = enc.encode(prefix)

    context_size = len(context_tokens)
    continuation_tokens = enc.encode(continuation)

    full_sentence = prefix+continuation

    logits = get_logits(sess, run_name, checkpoint_dir, model_name, model_dir, full_sentence, all=True)

    logits = logits[context_size-1:-1, :]  # only continuation logits
    logitmeans = np.mean(logits, axis=1)
    logits = logits - logitmeans[:, None]
    explogits = np.exp(logits)
    probabs = explogits / np.sum(explogits,axis=1)[:, None]
    
    probab_scores = np.nan_to_num([probabs[i, index] for i, index in enumerate(continuation_tokens)])
    perplexity = 2 ** (-np.mean(np.log2(probab_scores)))
    return perplexity