class OutputSection:
    def __init__(self) -> None:
        
        self.linear = nn.Linear(model_dimension, trg_vocab_size)
        self.log_softmax = nn.LogSoftmax(dim = -1)
    