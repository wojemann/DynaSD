class HFER(DynaSDBase):
    def __init__(self, w_size=1, w_stride=0.125):
        super().__init__(w_size=w_size, w_stride=w_stride)

    def fit(self, X):
        self.X = X

    def forward(self, X):
        return self.X