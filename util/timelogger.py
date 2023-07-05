import time


class TimeLogger:

    def logAfter(self, funcName) -> ():
        start = time.perf_counter()

        def after():
            timeElapsed = time.perf_counter() - start
            print(f'The function {funcName} needs {timeElapsed} seconds.')

        return after
