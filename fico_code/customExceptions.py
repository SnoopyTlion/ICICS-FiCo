import traceback


class NoTextException(Exception):
    def __init__(self, message):
        traceback.print_stack()
        self.message = message

    def __str__(self):
        return f"{self.message}"


class NoNeedExploreException(Exception):
    def __init__(self, message):
        traceback.print_stack()
        self.message = message

    def __str__(self):
        return f"{self.message}"


def foo():
    raise NoTextException('notextException')


if __name__ == "__main__":
    try:
        foo()
    except Exception as e:
        print(e)
