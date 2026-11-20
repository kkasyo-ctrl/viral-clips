"""OpenAI connector: ask the moments model for clip-worthy sentence ranges as structured output."""
import functools

from openai import OpenAI
from pydantic import BaseModel

import config

SYSTEM_PROMPT = """\
You are judging the transcript of a long-form video to find moments that can be cut out and posted as \
standalone short-form clips (TikTok, Reels, YouTube Shorts).

You have not heard the audio. You only have the words, so judge only what the words carry: do not assume \
tone of voice, delivery, laughter or visuals will rescue a line that is flat on the page.

The person who watches the clip has no prior context. They have not seen the rest of the video, do not know \
who is speaking, and will scroll away within two seconds if the opening does not grab them. A good moment:
- opens cold: its first sentence makes sense on its own and creates curiosity, tension or a strong claim. It \
does not answer a question we never hear or point back at something said earlier ("like I said", "that guy", \
"the second one");
- is self-contained: everything needed to follow it is inside the clip;
- has a payoff before it ends: a punchline, reveal, story resolution, surprising fact, strong opinion or \
concrete takeaway;
- runs roughly 20-75 seconds when spoken (use the timestamps).

The transcript has one sentence per line: `[id] (hh:mm:ss) speaker: text`. start_sentence and end_sentence are \
the inclusive ids of the first and last sentence of a moment; copy them exactly from the transcript. Moments \
must not overlap.

A separate scorer rates every moment you return, so include every moment that plausibly clears the bar rather \
than only the best few, but do not pad the list with moments that fail it. Return an empty list if nothing \
qualifies.

title: a short hook-style title for the clip (at most ~8 words).
reason: one sentence on why it works for a viewer with no context.
"""

GAMING_PROMPT = """\
You are judging the transcript of a gaming livestream to find moments that can be cut out and posted as \
standalone short-form clips (TikTok, Reels, YouTube Shorts). Each clip shows the streamer's facecam above the \
gameplay, with their words as captions.

You have not seen the video. You have the streamer's words, plus a marker where their voice spiked well above \
its usual level: `[LOUD +11dB]` at the end of a line means shouting, screaming, a laugh or a rage moment \
around that sentence; the bigger the number, the bigger the reaction. On a stream the reaction is often the \
moment: a scream, a sudden "NO WAY", a burst of laughter or a rant over something that just happened in the \
game. Use the words to tell what happened and the markers to tell how hard it hit. Quiet narration with \
nothing happening is rarely a clip, however well spoken.

The person who watches the clip has no prior context. They have not seen the rest of the stream, do not know \
the game's state or the in-jokes with chat, and will scroll away within two seconds if the opening does not \
grab them. A good moment:
- opens cold: within its first seconds something is clearly happening or about to (a chase, a fight, a risky \
play, a bold claim, a setup that promises a reaction). It does not lean on "like I said", earlier rounds or \
chat messages we never see;
- builds to a payoff the viewer can feel: the clutch, the fail, the jump scare, the glitch, the rage, the \
punchline -- usually where the LOUD markers are;
- is self-contained: someone who has never played the game can still tell what went wrong or right;
- runs roughly 15-60 seconds (use the timestamps); end shortly after the reaction, not minutes later.

The transcript has one sentence per line: `[id] (hh:mm:ss) speaker: text`, optionally followed by a loudness \
marker. start_sentence and end_sentence are the inclusive ids of the first and last sentence of a moment; copy \
them exactly from the transcript. Moments must not overlap.

A separate scorer rates every moment you return, so include every moment that plausibly clears the bar rather \
than only the best few, but do not pad the list with moments that fail it. Return an empty list if nothing \
qualifies.

title: a short hook-style title for the clip (at most ~8 words), in the casual voice of gaming clips.
reason: one sentence on why it works for a viewer with no context.
"""

PROMPTS = {"podcast": SYSTEM_PROMPT, "gaming": GAMING_PROMPT}

_SCHEMA = {
    "type": "object",
    "properties": {
        "moments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start_sentence": {"type": "integer", "description": "id of the first sentence (inclusive)"},
                    "end_sentence": {"type": "integer", "description": "id of the last sentence (inclusive)"},
                    "title": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["start_sentence", "end_sentence", "title", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["moments"],
    "additionalProperties": False,
}


class Moment(BaseModel):
    start_sentence: int
    end_sentence: int
    title: str
    reason: str


class _MomentList(BaseModel):
    moments: list[Moment]


@functools.cache
def _client() -> OpenAI:
    return OpenAI(timeout=1800, max_retries=3)  # reads OPENAI_API_KEY


def extract_moments(transcript: str, note: str | None = None, clip_format: str = "podcast") -> list[Moment]:
    extra = {"reasoning_effort": config.MOMENTS_REASONING_EFFORT} if config.MOMENTS_REASONING_EFFORT else {}
    response = _client().chat.completions.create(
        model=config.MOMENTS_MODEL,
        messages=[
            {"role": "system", "content": PROMPTS[clip_format]},
            {"role": "user", "content": f"{note}\n\n{transcript}" if note else transcript},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "moments", "strict": True, "schema": _SCHEMA},
        },
        **extra,
    )
    choice = response.choices[0]
    if choice.message.refusal:
        raise RuntimeError(f"{config.MOMENTS_MODEL} refused: {choice.message.refusal}")
    if choice.finish_reason == "length":
        raise RuntimeError(f"{config.MOMENTS_MODEL} ran out of output tokens before finishing the moment list")
    return _MomentList.model_validate_json(choice.message.content).moments
