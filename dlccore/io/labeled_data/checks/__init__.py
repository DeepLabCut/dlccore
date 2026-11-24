"""Pure checks on labeled data. Never a verdict.

A write refuses or warns on a check, and validation reports the same check as a finding; neither re-derives it.
``file`` holds facts about a file as stored, ``model`` facts about the labels it holds.

Checks are for acting at the moment of use, such as refusing a write; to know a file's condition, use validation,
whose findings carry every relevant check.
"""
