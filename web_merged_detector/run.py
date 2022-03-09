from app.app import app
import logging
from logging.handlers import TimedRotatingFileHandler


if __name__ == '__main__':
    app.debug = True
    handler = TimedRotatingFileHandler(
        'log/appcheck.log', when='D', interval=1, backupCount=15,
        encoding='UTF-8', delay=False, utc=True
    )
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(lineno)s - %(message)s'
    )
    handler.setFormatter(formatter)
    handler.setLevel(logging.INFO)
    app.logger.addHandler(handler)
    app.run(port=8080)

