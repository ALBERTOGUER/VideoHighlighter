"""DroneReel-AI: one continuous highlight shot out of one DJI video.

Built on VideoHighlighter, but deliberately apart from its pipeline: nothing in
this package imports ``pipeline``, ``main`` or the GUI, so it runs on a
Raspberry Pi with a handful of light dependencies (see ``requirements.txt``).
Upstream modules are reused only where their import chain is stdlib-only, and
``tests/test_isolation.py`` keeps it that way.
"""

__version__ = "0.1.0"
