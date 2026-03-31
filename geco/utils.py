#create a wrapper for the logging module
import logging
import os

def createLogger(name:str = 'mylogger') -> logging.Logger:
    """
    Create a logger with the specified name and set it up to log to a file. Formatter '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    :param name: the name of the logger to create. Defaults to 'mylogger'.
    :type name: str, optional

    :return: a logger instance with the specified name, set up to log to a file.
    :rtype: logging.Logger

    Example usage:
    >>> log = create('mylogger')
    >>> log.info('This is an info message.')
    2024-06-01 12:00:00,000 - mylogger - INFO - This is an info message.
    >>> log.error('This is an error message.')
    2024-06-01 12:00:01,000 - mylogger - ERROR - This is an error message.
    """
    try:   
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)

        log_dir = './logs'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        log_file = os.path.join(log_dir, f'{name}.log')
        open(log_file, 'w').close()
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)

        logger.addHandler(fh)

        logger.info('Logger successfully created.')
        return logger
    except Exception as e:
        print(f'Error creating logger: {e}')
        return None