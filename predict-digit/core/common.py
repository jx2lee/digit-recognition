def print_info(string):
    print('\033[31m \033[43m' + '[INFO]'+ '\033[0m' + ' ' + string)

def print_error(string):
    print('\033[37m \033[101m' + '[ERROR]'+ '\033[0m' + ' ' + string)

def make_logger(name=None):
    import logging
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - t%(levelname)s - %(message)s")

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    
    logger.addHandler(console)
    return logger
