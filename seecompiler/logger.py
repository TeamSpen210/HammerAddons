import logging

class LogMessage:
    """Allow using str.format() in logging messages.

    The __str__() method performs the joining.
    """
    def __init__(self, fmt, args, kwargs):
        self.fmt = fmt
        self.args = args
        self.kwargs = kwargs
        self.has_args = kwargs or args

    def format_msg(self):
        # Only format if we have arguments!
        # That way { or } can be used in regular messages.
        if self.has_args:
            f = self.fmt = str(self.fmt).format(*self.args, **self.kwargs)

            # Don't repeat the formatting
            del self.args, self.kwargs
            self.has_args = False
            return f
        else:
            return str(self.fmt)

    def __str__(self):
        """Format the string, and add an ASCII indent."""
        msg = self.format_msg()

        if '\n' not in msg:
            return msg

        # For multi-line messages, add an indent so they're associated
        # with the logging tag.
        lines = msg.split('\n')
        if lines[-1].isspace():
            # Strip last line if it's blank
            del lines[-1]
        # '|' beside all the lines, '|_ beside the last. Add an empty
        # line at the end.
        return '\n | '.join(lines[:-1]) + '\n |_' + lines[-1] + '\n'


class LoggerAdapter(logging.LoggerAdapter):
    """Fix loggers to use str.format().

    """
    def __init__(self, logger: logging.Logger, alias=None) -> None:
        # Alias is a replacement module name for log messages.
        self.alias = alias
        super(LoggerAdapter, self).__init__(logger, extra={})

    def log(self, level, msg, *args, exc_info=None, stack_info=False, **kwargs):
        """This version of .log() is for str.format() compatibility.

        The message is wrapped in a LogMessage object, which is given the
        args and kwargs
        """
        if self.isEnabledFor(level):
            self.logger._log(
                level,
                LogMessage(msg, args, kwargs),
                (), # No positional arguments, we do the formatting through
                # LogMessage..
                # Pull these two arguments out of kwargs, so they can be set..
                exc_info=exc_info,
                stack_info=stack_info,
                extra={'alias': self.alias},
            )


def init_logging(filename: str=None, main_logger='', on_error=None) -> logging.Logger:
    """Setup the logger and logging handlers.

    If filename is set, all logs will be written to this file as well.
    This also sets sys.except_hook, so uncaught exceptions are captured.
    on_error should be a function to call when this is done
    (taking type, value, traceback).
    """
    global short_log_format, long_log_format
    global stderr_loghandler, stdout_loghandler
    import logging
    from logging import handlers
    import sys, io, os

    class NewLogRecord(logging.getLogRecordFactory()):
        """Allow passing an alias for log modules."""
        # This breaks %-formatting, so only set when init_logging() is called.

        alias = None  # type: str

        def getMessage(self):
            """We have to hook here to change the value of .module.

            It's called just before the formatting call is made.
            """
            if self.alias is not None:
                self.module = self.alias
            return str(self.msg)
    logging.setLogRecordFactory(NewLogRecord)

    logger = logging.getLogger('SEE')
    logger.setLevel(logging.DEBUG)

    # Put more info in the log file, since it's not onscreen.
    long_log_format = logging.Formatter(
        '[{levelname}] {module}.{funcName}(): {message}',
        style='{',
    )
    # Console messages, etc.
    short_log_format = logging.Formatter(
        # One letter for level name
        '[{levelname[0]}] {module}: {message}',
        style='{',
    )

    if filename is not None:
        # Make the directories the logs are in, if needed.
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # The log contains DEBUG and above logs.
        # We rotate through logs of 500kb each, so it doesn't increase too much.
        log_handler = handlers.RotatingFileHandler(
            filename,
            maxBytes=500 * 1024,
            backupCount=1,
        )
        log_handler.setLevel(logging.DEBUG)
        log_handler.setFormatter(long_log_format)
        logger.addHandler(log_handler)

        err_log_handler = handlers.RotatingFileHandler(
            filename[:-3] + 'error.' + filename[-3:],
            maxBytes=500 * 1024,
            backupCount=1,
        )
        err_log_handler.setLevel(logging.WARNING)
        err_log_handler.setFormatter(long_log_format)

        logger.addHandler(err_log_handler)

    # This is needed for multiprocessing, since it tries to flush stdout.
    # That'll fail if it is None.
    class NullStream(io.IOBase):
        """A stream object that discards all data."""
        def __init__(self):
            super(NullStream, self).__init__()

        @staticmethod
        def write(self, *args, **kwargs):
            pass

        @staticmethod
        def read(*args, **kwargs):
            return ''

    if sys.stdout:
        stdout_loghandler = logging.StreamHandler(sys.stdout)
        stdout_loghandler.setLevel(logging.INFO)
        stdout_loghandler.setFormatter(long_log_format)
        logger.addHandler(stdout_loghandler)

        if sys.stderr:
            def ignore_warnings(record: logging.LogRecord):
                """Filter out messages higher than WARNING.

                Those are handled by stdError, and we don't want duplicates.
                """
                return record.levelno < logging.WARNING
            stdout_loghandler.addFilter(ignore_warnings)
    else:
        sys.stdout = NullStream()

    if sys.stderr:
        stderr_loghandler = logging.StreamHandler(sys.stderr)
        stderr_loghandler.setLevel(logging.WARNING)
        stderr_loghandler.setFormatter(long_log_format)
        logger.addHandler(stderr_loghandler)
    else:
        sys.stderr = NullStream()

    # Use the exception hook to report uncaught exceptions, and finalise the
    # logging system.
    old_except_handler = sys.excepthook

    def except_handler(exc_type, exc_value, exc_tb):
        """Log uncaught exceptions."""
        if not issubclass(exc_type, Exception):
            # It's subclassing BaseException (KeyboardInterrupt, SystemExit),
            # so we should quit without messages.
            logging.shutdown()
            return

        logger._log(
            level=logging.ERROR,
            msg='Uncaught Exception:',
            args=(),
            exc_info=(exc_type, exc_value, exc_tb),
        )
        logging.shutdown()
        if on_error is not None:
            on_error(exc_type, exc_value, exc_tb)
        # Call the original handler - that prints to the normal console.
        old_except_handler(exc_type, exc_value, exc_tb)

    sys.excepthook = except_handler

    if main_logger:
        return get_logger(main_logger)
    else:
        return LoggerAdapter(logger)


def get_logger(name: str='', alias: str=None) -> logging.Logger:
    """Get the named logger object.

    This puts the logger into the BEE2 namespace, and wraps it to
    use str.format() instead of % formatting.
    If set, alias is the name to show for the module.
    """
    if name:
        return LoggerAdapter(logging.getLogger('SEE.' + name), alias)
    else:  # Allow retrieving the main logger.
        return LoggerAdapter(logging.getLogger('SEE'), alias)
