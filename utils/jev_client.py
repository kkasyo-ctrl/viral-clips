"""Jev (TypeSafe) connector: calibrated yes/no probabilities for each candidate clip.

One system_one call per clip asks four Noul hypotheses at once; calls run concurrently. A gaming
clip is asked the same four questions, told it comes from a stream and shown where the streamer's
voice spiked, since there the reaction carries as much as the words.
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
_GAMING_CONTEXT = (
    "The state is the transcript of a clip cut from a gaming livestream: the streamer's commentary over their "
    "gameplay, posted on its own as a short-form vertical video (TikTok, Reels, YouTube Shorts) with the facecam "
    "above the game, to viewers who have never seen the stream. loud_moments lists where in the clip the "
    "streamer's voice spiked well above their usual level (shouting, screaming, laughing, raging), in dB over it."
)
_CARRIES = {
    "podcast": "and that comes through in the words themselves.",
    "gaming": "and that comes through in the streamer's words and how hard they react.",
}


def _hypotheses(context: str, carries: str) -> dict[str, Noul]:
    return {
        "p_viral": Noul(
            instructions=f"{context} Will this clip go viral: hold viewers to the end and get shared, commented on "
            "or rewatched far more than a typical clip?",
            criteria={
                "true": "A cold viewer would stop scrolling within the first seconds and watch to the end: the clip is "
                f"surprising, funny, emotional, controversial or unusually useful, {carries}",
                "false": "The clip is ordinary, slow, low-stakes or needs outside context, or its interesting part never "
                "arrives; most viewers would scroll past.",
            },
        ),
        "p_self_contained": Noul(
            instructions=f"{context} Can a viewer with no prior context fully follow this clip?",
            criteria={
                "true": "Everything needed to understand it is inside the clip: who or what is being discussed is clear "
                "and nothing depends on earlier parts of the conversation.",
                "false": "It relies on earlier context: unexplained references (\"that guy\", \"like I said\", \"the "
                "second one\"), an answer to a question we never hear, or a thought that starts or ends midway.",
            },
        ),
        "p_has_payoff": Noul(
            instructions=f"{context} Does the clip deliver a payoff before it ends?",
            criteria={
                "true": "It lands a clear payoff inside the clip: a punchline, reveal, story resolution, surprising fact, "
                "strong opinion or concrete takeaway.",
                "false": "It builds up without landing, trails off, or ends before the point is made.",
            },
        ),
        "p_opens_cold": Noul(
            instructions=f"{context} Does the clip's first sentence work as a cold open for someone who just "
            "scrolled to it?",
            criteria={
                "true": "The first sentence makes sense on its own and hooks immediately with a claim, question, "
                "tension or vivid detail.",
                "false": "It opens with filler, slow setup, a reply to something unseen, or a reference to earlier "
                "context.",
            },
        ),
    }


HYPOTHESES = _hypotheses(_CONTEXT, _CARRIES["podcast"])
GAMING_HYPOTHESES = _hypotheses(_GAMING_CONTEXT, _CARRIES["gaming"])
_BY_FORMAT = {"podcast": HYPOTHESES, "gaming": GAMING_HYPOTHESES}


def score_clips(states: list[dict], clip_format: str = "podcast") -> list[dict[str, float] | None]:
    """Probabilities per clip, in input order; None where a call failed even after retries."""
    if not states:
        return []
    results = asyncio.run(_score_all(states, _BY_FORMAT[clip_format]))
    failures = [r for r in results if isinstance(r, BaseException)]
    if failures and len(failures) == len(results):
        raise failures[0]
    for failure in failures:
        log.warning("Jev scoring failed for one clip: %s", failure)
    return [None if isinstance(r, BaseException) else r for r in results]


async def _score_all(states: list[dict], questions: dict[str, Noul]) -> list:
    semaphore = asyncio.Semaphore(config.JEV_CONCURRENCY)
    # reads TYPESAFE_API_KEY
    async with AsyncTypeSafeClient(model=config.JEV_MODEL, retry=RetryPolicy(max_retries=4)) as client:

        async def score(state: dict) -> dict[str, float]:
            async with semaphore:
                response = await client.system_one(state=state, questions=questions)
            return {name: response.answers[name].noul for name in questions}

        return await asyncio.gather(*(score(s) for s in states), return_exceptions=True)
