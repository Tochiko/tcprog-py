import time


class NoneLogger:

    def logAfter(self, funcName) -> ():

        def after():
            pass

        return after
