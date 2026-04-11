"""Scheduled check-ins and sensor-triggered conversations."""

import schedule
import time
import yaml

with open("config.yaml") as f:
    _cfg = yaml.safe_load(f)


def start(on_trigger):
    """Register all scheduled messages and run the loop."""
    for item in _cfg.get("schedule", []):
        schedule.every().day.at(item["time"]).do(on_trigger, item["message"])

    for item in _cfg.get("reminders", []):
        schedule.every().day.at(item["time"]).do(on_trigger, item["message"])

    while True:
        schedule.run_pending()
        time.sleep(30)
