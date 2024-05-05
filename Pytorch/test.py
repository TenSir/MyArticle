class SeqDataLoader:
    """An iterator to load sequence data."""
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        """两种加载数据的形式`"""
        if use_random_iter:
            self.data_iter_fn = 1
        else:
            self.data_iter_fn = 2
        self.corpus, self.vocab = 3,4
        self.batch_size, self.num_steps = batch_size, num_steps


data_iter = SeqDataLoader(11, 111, use_random_iter=False, max_tokens = 1111)
print(data_iter)
print(data_iter.batch_size)