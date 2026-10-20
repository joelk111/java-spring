"""Exception hierarchy. Each class carries the CLI exit code from plan §8."""


class KitsplitError(Exception):
    exit_code = 10


class KitsplitInputError(KitsplitError):
    """Bad input file or bad arguments."""

    exit_code = 2


class KitsplitModelError(KitsplitError):
    """Model missing, unknown, or corrupt (sha256 mismatch)."""

    exit_code = 3


class KitsplitDeviceError(KitsplitError):
    """Requested compute device is unavailable."""

    exit_code = 4


class KitsplitOutputExistsError(KitsplitError):
    """Output exists and --overwrite was not given."""

    exit_code = 5


class KitsplitLicenseError(KitsplitError):
    """A dataset/weight license forbids the requested use."""

    exit_code = 6
