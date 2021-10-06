class Data_Not_Found_Error(RuntimeError):
    def __init__(self, typee, beta, n = None):
        super().__init__(
            ("Requested data not found in disk or RAM. type: %s, beta: %s, n = %d" % (typee, beta, n)) if n else
            ("Requested data not found in disk or RAM. type: %s, beta: %s" % (typee, beta))
        )