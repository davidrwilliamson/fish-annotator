class Error(Exception):
    """Base class for exceptions in this module."""
    pass


class NoImageFilesError(Error):
    """Exception raised when a folder contains no silc images.

    Attributes:
        folder -- folder being checked
        message -- explanation of error
    """

    def __init__(self, folder):
        self.message = "NoImageFilesError: No silc images found in folder {}".format(folder)
        super().__init__(self.message)
