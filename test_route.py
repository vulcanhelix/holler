"""Offline tests for route(). The Cerebras call is mocked; no network, no API key."""

import json

import pytest

from holler.route import route

ALIASES = """\
[sites]
warmy = "https://app.warmy.io"
heyreach = "https://app.heyreach.io"
instantly = "https://app.instantly.ai"
github = "https://github.com"
google = "https://www.google.com"
webuycars = "https://www.webuycars.co.za"

[mishearings]
"we buy cars" = "webuycars"
wormy = "warmy"
"warm e" = "warmy"
"""


@pytest.fixture()
def aliases_path(tmp_path):
    p = tmp_path / "aliases.toml"
    p.write_text(ALIASES)
    return p


def fake_llm(mapping):
    """A call_llm(system, user) that answers by substring of the normalized transcript."""

    def call(system, user):
        heard = json.loads(user)["heard"]
        for key, out in mapping.items():
            if key in heard:
                return dict(out)
        raise AssertionError(f"no fake response for heard={heard!r}")

    return call


def ok(url, *goals):
    return {"url": url, "goals": list(goals)}


def test_named_site_two_goals(aliases_path):
    llm = fake_llm({"warmy": ok("https://app.warmy.io", "open the inboxes page", "read inbox 14 status")})
    task = route("open warmy and check inbox fourteen", call_llm=llm, aliases_path=aliases_path)
    assert task == {"url": "https://app.warmy.io", "goals": ["open the inboxes page", "read inbox 14 status"]}


def test_named_site_one_goal(aliases_path):
    llm = fake_llm({"heyreach": ok("https://app.heyreach.io", "read the campaign stats")})
    task = route("go to heyreach and read the campaign stats", call_llm=llm, aliases_path=aliases_path)
    assert task["url"] == "https://app.heyreach.io"
    assert len(task["goals"]) == 1


def test_mishearing_wormy(aliases_path):
    llm = fake_llm({"warmy": ok("https://app.warmy.io", "open the inboxes page", "read inbox 14 status")})
    task = route("open wormy and check inbox fourteen", call_llm=llm, aliases_path=aliases_path)
    assert task["url"] == "https://app.warmy.io"


def test_mishearing_warm_e(aliases_path):
    llm = fake_llm({"warmy": ok("https://app.warmy.io", "open the inboxes page", "read inbox 7 status")})
    task = route("warm e check inbox seven", call_llm=llm, aliases_path=aliases_path)
    assert task["url"] == "https://app.warmy.io"


def test_well_known_site(aliases_path):
    llm = fake_llm({"github": ok("https://github.com/notifications", "read the notifications")})
    task = route("check my github notifications", call_llm=llm, aliases_path=aliases_path)
    assert task["url"] == "https://github.com"  # alias wins over the model's deeper URL


def test_unaliased_site(aliases_path):
    llm = fake_llm({"wikipedia": ok("https://en.wikipedia.org", "open the article about godel")})
    task = route("open wikipedia and find the godel article", call_llm=llm, aliases_path=aliases_path)
    assert task["url"] == "https://en.wikipedia.org"


def test_model_url_overridden_for_named_site(aliases_path):
    llm = fake_llm({"instantly": ok("https://instantly.ai.evil.example", "read the warmup status")})
    task = route("open instantly and show warmup status", call_llm=llm, aliases_path=aliases_path)
    assert task["url"] == "https://app.instantly.ai"


def test_explicit_destination_wins_over_alias(aliases_path):
    # "search we buy cars on google" — model chose google; the named alias must not override it.
    llm = fake_llm({"webuycars": ok("https://www.google.com", "search for webuycars", "open the first result")})
    task = route("search we buy cars on google and open the first result", call_llm=llm, aliases_path=aliases_path)
    assert task["url"] == "https://www.google.com"


def test_bad_url_returns_none(aliases_path):
    # No alias named, so the model's malformed URL reaches validation untouched.
    llm = fake_llm({"blorp": ok("not-a-url", "open the dashboard")})
    assert route("open blorp", call_llm=llm, aliases_path=aliases_path) is None


def test_empty_goals_returns_none(aliases_path):
    llm = fake_llm({"warmy": {"url": "https://app.warmy.io", "goals": []}})
    assert route("open warmy", call_llm=llm, aliases_path=aliases_path) is None


def test_long_goal_returns_none(aliases_path):
    llm = fake_llm({"warmy": ok("https://app.warmy.io", "x" * 120)})
    assert route("open warmy", call_llm=llm, aliases_path=aliases_path) is None


def test_garbage_json_returns_none(aliases_path):
    llm = fake_llm({"warmy": {"nope": True}})
    assert route("open warmy", call_llm=llm, aliases_path=aliases_path) is None
