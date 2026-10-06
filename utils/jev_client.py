"""Jev (TypeSafe) connector: calibrated yes/no probabilities for each candidate clip.

One system_one call per clip asks four Noul hypotheses at once; calls run concurrently.
"""
import asyncio
import logging

from typesafe_sdk import AsyncTypeSafeClient, Noul, RetryPolicy

import config

log = logging.getLogger(__name__)

_CONTEXT = (
    "The state is the transcript of a clip cut from a longer video, to be posted on its own as a short-form "
    "vertical video (TikTok, Reels, YouTube Shorts) to viewers who have never seen the source video."
)

HYPOTHESES = {
    "p_viral": Noul(
        instructions=f"{_CONTEXT} Will this clip go viral: hold viewers to the end and get shared, commented on "
        "or rewatched far more than a typical clip?",
        criteria={
            "true": "A cold viewer would stop scrolling within the first seconds and watch to the end: the clip is "
            "surprising, funny, emotional, controversial or unusually useful, and that comes through in the words "
            "themselves.",
            "false": "The clip is ordinary, slow, low-stakes or needs outside context, or its interesting part never "
            "arrives; most viewers would scroll past.",
        },
    ),
    "p_self_contained": Noul(
        instructions=f"{_CONTEXT} Can a viewer with no prior context fully follow this clip?",
        criteria={
            "true": "Everything needed to understand it is inside the clip: who or what is being discussed is clear "
            "and nothing depends on earlier parts of the conversation.",
            "false": "It relies on earlier context: unexplained references (\"that guy\", \"like I said\", \"the "
            "second one\"), an answer to a question we never hear, or a thought that starts or ends midway.",
        },
    ),
    "p_has_payoff": Noul(
        instructions=f"{_CONTEXT} Does the clip deliver a payoff before it ends?",
        criteria={
            "true": "It lands a clear payoff inside the clip: a punchline, reveal, story resolution, surprising fact, "
            "strong opinion or concrete takeaway.",
            "false": "It builds up without landing, trails off, or ends before the point is made.",
        },
    ),
    "p_opens_cold": Noul(
        instructions=f"{_CONTEXT} Does the clip's first sentence work as a cold open for someone who just "
        "scrolled to it?",
        criteria={
            "true": "The first sentence makes sense on its own and hooks immediately with a claim, question, "
            "tension or vivid detail.",
            "false": "It opens with filler, slow setup, a reply to something unseen, or a reference to earlier "
            "context.",
        },
    ),
}


def score_clips(states: list[dict]) -> list[dict[str, float] | None]:
    """Probabilities per clip, in input order; None where a call failed even after retries."""
    if not states:
        return []
    results = asyncio.run(_score_all(states))
    failures = [r for r in results if isinstance(r, BaseException)]
    if failures and len(failures) == len(results):
        raise failures[0]
    for failure in failures:
        log.warning("Jev scoring failed for one clip: %s", failure)
    return [None if isinstance(r, BaseException) else r for r in results]


async def _score_all(states: list[dict]) -> list:
    semaphore = asyncio.Semaphore(config.JEV_CONCURRENCY)
    # reads TYPESAFE_API_KEY
    async with AsyncTypeSafeClient(model=config.JEV_MODEL, retry=RetryPolicy(max_retries=4)) as client:

        async def score(state: dict) -> dict[str, float]:
            async with semaphore:
                response = await client.system_one(state=state, questions=HYPOTHESES)
            return {name: response.answers[name].noul for name in HYPOTHESES}

        return await asyncio.gather(*(score(s) for s in states), return_exceptions=True)
