import os
from typing import Optional
from abc import ABC, abstractmethod

import requests
from loguru import logger

from common.utils import singleton


class NotifierBase(ABC):
    @abstractmethod
    def notify(self, message: str) -> Optional[Exception]:
        raise NotImplementedError

    @abstractmethod
    def is_valid(self):
        raise NotImplementedError


@singleton
class BarkNotifier(NotifierBase):
    valid: bool

    def __init__(self, notification_key: str):
        super().__init__()
        if notification_key != "" and notification_key is not None:
            self.url_pattern = "https://api.day.app/" + notification_key + "/{}?isArchive=1"
            self.session = requests.session()
        else:
            self.url_pattern = None
            self.session = None

    @staticmethod
    def get_instance():
        return BarkNotifier()

    @property
    def is_valid(self):
        return self.url_pattern is not None

    def notify(self, message: str) -> Optional[Exception]:
        if self.url_pattern is not None:
            try:
                self.session.post(self.url_pattern.format(f"[UniFolding] {message}"))
            except Exception as e:
                logger.error(f'Failed to connect to notification server: {e}')
                return e
        else:
            logger.warning("BarkNotifier is not initialized.")
            return Exception("BarkNotifier is not initialized.")


_bark_notifier: Optional[BarkNotifier] = None


def get_bark_notifier(recreate: bool = False) -> BarkNotifier:
    global _bark_notifier
    if _bark_notifier is None or recreate:
        _bark_notifier = BarkNotifier(os.environ.get("CONFIG_BARK_NOTIFICATION_KEY", None))

    return _bark_notifier


@singleton
class SlackNotifier(NotifierBase):
    def __init__(self, webhook_url: str):
        super().__init__()
        if webhook_url != "" and webhook_url is not None:
            self.webhook_url = webhook_url
            self.session = requests.session()
        else:
            self.webhook_url = None
            self.session = None

    @staticmethod
    def get_instance():
        return SlackNotifier()

    @property
    def is_valid(self):
        return self.webhook_url is not None

    def notify(self, message: str) -> Optional[Exception]:
        if self.webhook_url is None:
            logger.debug(f"No webhook url provided!")
            return Exception("No webhook url provided!")
        else:
            resp = requests.post(self.webhook_url, json={'text': message})
            if resp.status_code != 200:
                logger.debug(f"Failed to send message to slack! Status code: {resp.status_code}")
                return Exception(f"Failed to send message to slack! Status code: {resp.status_code}")
            else:
                return None


_slack_notifier: Optional[SlackNotifier] = None


def get_slack_notifier(recreate: bool = False) -> SlackNotifier:
    global _slack_notifier
    if _slack_notifier is None or recreate:
        _slack_notifier = SlackNotifier(os.environ.get("CONFIG_SLACK_WEBHOOK_URL", None))
    return _slack_notifier


if __name__ == '__main__':
    #
    os.environ[
        "CONFIG_SLACK_WEBHOOK_URL"
    ] = "https://hooks.slack.com/services/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"

    os.environ[
        "CONFIG_BARK_NOTIFICATION_KEY"
    ] = ("cwhM24wAEDwa4ReUnnWsXf")
    # Test
    notifier = get_bark_notifier()
    notifier.notify("Test message")
    notifier = get_slack_notifier()
    notifier.notify("Test message")
