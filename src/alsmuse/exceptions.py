"""Custom exceptions for ALSmuse.

This module defines the exception hierarchy for handling errors
during ALS file parsing and analysis.
"""


class ALSmuseError(Exception):
    """Base exception for ALSmuse.

    All custom exceptions in this library inherit from this class,
    allowing callers to catch all ALSmuse-related errors with a
    single except clause.
    """


class ParseError(ALSmuseError):
    """Error parsing an ALS file.

    Raised when the ALS file cannot be read, decompressed,
    or parsed as valid XML.
    """


class TrackNotFoundError(ALSmuseError):
    """Specified track not found in LiveSet.

    Raised when attempting to find a track by name that
    does not exist in the parsed LiveSet.
    """


class AlignmentError(ALSmuseError):
    """Raised when lyrics alignment fails.

    This can occur when:
    - The stable-ts package is not installed
    - Audio files cannot be loaded
    - The alignment model fails to process the audio
    - No valid words are produced from alignment
    """
