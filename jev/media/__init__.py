"""JEV media: narrated explainer videos, editing your own footage, slides, images, local speech
and optional local generative models. Everything runs on this machine.

    jev.media.common      errors, workspace checks, limits, fonts
    jev.media.ffmpeg      locating and running FFmpeg safely (argument lists, timeouts, caps)
    jev.media.tts         local text-to-speech (Piper, espeak-ng, macOS say, Windows SAPI, silence)
    jev.media.slides      16:9 slide rendering with Pillow (title, bullets, code, image, equation, chart)
    jev.media.images      charts, box-and-arrow diagrams, title cards, contact sheets, image edits
    jev.media.video       explainer assembly: scenes -> speech -> clips -> crossfades -> captions -> MP4
    jev.media.footage     shot detection, semantic search, highlight reels and cut lists for your footage
    jev.media.generative  optional diffusers adapters (text-to-image, text-to-video), approval-gated

See docs/jev/media.md. The tools the agent calls are in jev.tools.media.
"""
