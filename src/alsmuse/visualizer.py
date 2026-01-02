"""Lyrics/cues video visualizer for ALSmuse.

Generates an MP4 video visualization of phrases with lyrics, section markers,
track events, and a progress bar. Uses Pillow for frame rendering and ffmpeg
for video encoding.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Generator
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .models import Phrase, TrackEvent

logger = logging.getLogger(__name__)

# Video specifications
VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720
FRAME_RATE = 24
VIDEO_CRF = 23

# Colors
BACKGROUND_COLOR = (0, 0, 0)
SECTION_COLOR = (136, 136, 136)  # #888888
TIMECODE_COLOR = (102, 102, 102)  # #666666
PREV_NEXT_LYRIC_COLOR = (68, 68, 68)  # #444444
CURRENT_LYRIC_COLOR = (255, 255, 255)  # #FFFFFF
PROGRESS_BAR_BG_COLOR = (51, 51, 51)  # #333333
PROGRESS_BAR_FILL_COLOR = (255, 255, 255)  # #FFFFFF

# Event colors by category
CATEGORY_COLORS: dict[str, tuple[int, int, int]] = {
    "drums": (229, 115, 115),  # #E57373 coral red
    "bass": (129, 199, 132),  # #81C784 soft green
    "vocals": (100, 181, 246),  # #64B5F6 sky blue
    "keys": (255, 213, 79),  # #FFD54F amber
    "guitar": (186, 104, 200),  # #BA68C8 purple
    "pad": (77, 208, 225),  # #4DD0E1 cyan
    "synth": (77, 208, 225),  # #4DD0E1 cyan (same as pad)
    "fx": (144, 164, 174),  # #90A4AE blue-grey
    "lead": (144, 164, 174),  # #90A4AE blue-grey
    "other": (144, 164, 174),  # #90A4AE blue-grey
}

# Typography sizes (approximate - will be scaled for the font)
SECTION_FONT_SIZE = 28
TIMECODE_FONT_SIZE = 24
PREV_NEXT_LYRIC_FONT_SIZE = 32
CURRENT_LYRIC_FONT_SIZE = 42
EVENT_BADGE_FONT_SIZE = 18

# Layout positions
SECTION_Y = 60
TIMECODE_Y = 60
LYRIC_CENTER_Y = 360
LYRIC_LINE_SPACING = 60
EVENT_ROW_Y = 580
PROGRESS_BAR_X = 40
PROGRESS_BAR_Y = 680
PROGRESS_BAR_WIDTH = 1200
PROGRESS_BAR_HEIGHT = 8

# Animation timing
EVENT_FADE_SECONDS = 4.0


@dataclass
class FontSet:
    """Collection of fonts for rendering at different sizes."""

    section: ImageFont.FreeTypeFont | ImageFont.ImageFont
    timecode: ImageFont.FreeTypeFont | ImageFont.ImageFont
    prev_next: ImageFont.FreeTypeFont | ImageFont.ImageFont
    current: ImageFont.FreeTypeFont | ImageFont.ImageFont
    current_bold: ImageFont.FreeTypeFont | ImageFont.ImageFont
    event: ImageFont.FreeTypeFont | ImageFont.ImageFont

    @classmethod
    def load(cls, font_path: Path | None = None) -> FontSet:
        """Load fonts for rendering.

        Tries to load a bundled font or system fallback.

        Args:
            font_path: Optional path to a specific TTF font file.

        Returns:
            FontSet with all required font sizes.
        """
        # Try to find a suitable font
        font_paths_to_try = []

        if font_path is not None:
            font_paths_to_try.append(font_path)

        # Common system font locations
        font_paths_to_try.extend(
            [
                # macOS
                Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
                Path("/System/Library/Fonts/Helvetica.ttc"),
                Path("/Library/Fonts/Arial.ttf"),
                # Linux
                Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
                Path("/usr/share/fonts/TTF/DejaVuSans.ttf"),
                Path("/usr/share/fonts/truetype/freefont/FreeSans.ttf"),
                # Windows
                Path("C:/Windows/Fonts/arial.ttf"),
            ]
        )

        loaded_font: Path | None = None
        for path in font_paths_to_try:
            if path.exists():
                loaded_font = path
                break

        if loaded_font is None:
            logger.warning("No suitable font found, using PIL default")

        def load_font_at_size(
            size: int,
        ) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
            if loaded_font is not None:
                return ImageFont.truetype(str(loaded_font), size)
            # Fall back to PIL default - this returns a basic font
            # Use load_default with explicit size if available (Pillow 10+)
            try:
                return ImageFont.load_default(size=size)
            except TypeError:
                # Older Pillow without size parameter
                return ImageFont.load_default()

        return cls(
            section=load_font_at_size(SECTION_FONT_SIZE),
            timecode=load_font_at_size(TIMECODE_FONT_SIZE),
            prev_next=load_font_at_size(PREV_NEXT_LYRIC_FONT_SIZE),
            current=load_font_at_size(CURRENT_LYRIC_FONT_SIZE),
            current_bold=load_font_at_size(CURRENT_LYRIC_FONT_SIZE),
            event=load_font_at_size(EVENT_BADGE_FONT_SIZE),
        )


@dataclass
class FrameState:
    """State for rendering a single frame.

    Captures all the information needed to render one frame of the video,
    including the current phrase, surrounding lyrics, active events, and timing.
    """

    current_time: float  # Current time in seconds
    total_time: float  # Total duration in seconds
    section_name: str  # Current section name
    prev_lyric: str  # Previous phrase lyric (dim)
    current_lyric: str  # Current phrase lyric (bright)
    next_lyric: str  # Next phrase lyric (dim)
    active_events: list[tuple[TrackEvent, float]]  # Events with age in seconds


def format_timecode(seconds: float) -> str:
    """Format seconds as MM:SS.

    Args:
        seconds: Time in seconds.

    Returns:
        Formatted time string in MM:SS format.
    """
    minutes = int(seconds) // 60
    secs = int(seconds) % 60
    return f"{minutes:02d}:{secs:02d}"


def get_category_color(category: str) -> tuple[int, int, int]:
    """Get the display color for a track category.

    Args:
        category: Track category name.

    Returns:
        RGB color tuple.
    """
    return CATEGORY_COLORS.get(category.lower(), CATEGORY_COLORS["other"])


def format_event_badge(event: TrackEvent) -> str:
    """Format a track event as a display badge.

    Args:
        event: TrackEvent to format.

    Returns:
        Badge string like "+DRUMS" or "-BASS".
    """
    prefix = "+" if event.event_type == "enter" else "-"
    # Use category as the display name for consistency
    return f"{prefix}{event.category.upper()}"


def compute_phrase_times(phrases: list[Phrase], bpm: float) -> list[tuple[float, float]]:
    """Compute start and end times for each phrase.

    Args:
        phrases: List of phrases.
        bpm: Tempo in beats per minute.

    Returns:
        List of (start_seconds, end_seconds) tuples for each phrase.
    """
    return [(p.start_time(bpm), p.start_time(bpm) + p.duration_seconds(bpm)) for p in phrases]


def build_frame_states(
    phrases: list[Phrase],
    bpm: float,
    total_beats: float,
) -> Generator[FrameState, None, None]:
    """Generate frame states for each frame of the video.

    Creates a FrameState for each frame based on the current time,
    determining which phrase is active, which events are visible, etc.

    Args:
        phrases: List of phrases with lyrics and events.
        bpm: Tempo in beats per minute.
        total_beats: Total song duration in beats.

    Yields:
        FrameState for each frame.
    """
    total_time = total_beats * 60 / bpm
    phrase_times = compute_phrase_times(phrases, bpm)
    total_frames = int(total_time * FRAME_RATE)

    # Track active events with their start times
    active_events: list[tuple[TrackEvent, float]] = []
    # Track which phrases we've already processed for event activation
    activated_phrase_indices: set[int] = set()
    prev_idx = -1

    for frame_num in range(total_frames):
        current_time = frame_num / FRAME_RATE

        # Find current phrase index
        current_idx = 0
        for i, (start, end) in enumerate(phrase_times):
            if start <= current_time < end:
                current_idx = i
                break
            elif current_time >= end:
                current_idx = i

        current_phrase = phrases[current_idx]

        # Get prev/next lyrics
        prev_lyric = ""
        if current_idx > 0:
            prev_lyric = phrases[current_idx - 1].lyric or ""

        next_lyric = ""
        if current_idx < len(phrases) - 1:
            next_lyric = phrases[current_idx + 1].lyric or ""

        # Add events from current phrase when we first enter it
        if current_idx != prev_idx and current_idx not in activated_phrase_indices:
            activated_phrase_indices.add(current_idx)
            for event in current_phrase.events:
                # Add all events from this phrase
                active_events.append((event, current_time))
        prev_idx = current_idx

        # Remove expired events (older than fade duration)
        active_events = [(e, t) for e, t in active_events if current_time - t < EVENT_FADE_SECONDS]

        # Compute event ages
        events_with_age = [(e, current_time - t) for e, t in active_events]

        yield FrameState(
            current_time=current_time,
            total_time=total_time,
            section_name=current_phrase.section_name,
            prev_lyric=prev_lyric,
            current_lyric=current_phrase.lyric or "",
            next_lyric=next_lyric,
            active_events=events_with_age,
        )


def render_frame(state: FrameState, fonts: FontSet) -> Image.Image:
    """Render a single frame of the visualization.

    Args:
        state: FrameState containing all data for this frame.
        fonts: FontSet with loaded fonts.

    Returns:
        PIL Image for this frame.
    """
    img = Image.new("RGB", (VIDEO_WIDTH, VIDEO_HEIGHT), BACKGROUND_COLOR)
    draw = ImageDraw.Draw(img)

    # Draw section name (upper left)
    section_text = state.section_name.upper()
    draw.text((40, SECTION_Y), section_text, font=fonts.section, fill=SECTION_COLOR)

    # Draw underline below section name
    section_bbox = draw.textbbox((40, SECTION_Y), section_text, font=fonts.section)
    underline_y = section_bbox[3] + 5
    draw.line([(40, underline_y), (section_bbox[2], underline_y)], fill=SECTION_COLOR, width=1)

    # Draw timecode (upper right)
    timecode = f"{format_timecode(state.current_time)} / {format_timecode(state.total_time)}"
    timecode_bbox = draw.textbbox((0, 0), timecode, font=fonts.timecode)
    timecode_width = timecode_bbox[2] - timecode_bbox[0]
    draw.text(
        (VIDEO_WIDTH - 40 - timecode_width, TIMECODE_Y),
        timecode,
        font=fonts.timecode,
        fill=TIMECODE_COLOR,
    )

    # Draw lyrics (centered vertically)
    # Previous lyric (dim, above current)
    if state.prev_lyric:
        _draw_centered_text(
            draw,
            state.prev_lyric,
            LYRIC_CENTER_Y - LYRIC_LINE_SPACING,
            fonts.prev_next,
            PREV_NEXT_LYRIC_COLOR,
        )

    # Current lyric (bright, center)
    if state.current_lyric:
        _draw_centered_text(
            draw,
            state.current_lyric.upper(),
            LYRIC_CENTER_Y,
            fonts.current_bold,
            CURRENT_LYRIC_COLOR,
        )

    # Next lyric (dim, below current)
    if state.next_lyric:
        _draw_centered_text(
            draw,
            state.next_lyric,
            LYRIC_CENTER_Y + LYRIC_LINE_SPACING,
            fonts.prev_next,
            PREV_NEXT_LYRIC_COLOR,
        )

    # Draw event badges (horizontal row near bottom)
    if state.active_events:
        _draw_event_badges(draw, state.active_events, fonts.event)

    # Draw progress bar
    progress = state.current_time / state.total_time if state.total_time > 0 else 0
    _draw_progress_bar(draw, progress)

    return img


def _draw_centered_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    y: int,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    color: tuple[int, int, int],
) -> None:
    """Draw text horizontally centered on the image.

    Handles multi-line text by splitting and centering each line.

    Args:
        draw: ImageDraw object to draw on.
        text: Text to draw.
        y: Vertical center position.
        font: Font to use.
        color: RGB color tuple.
    """
    lines = text.split("\n")
    line_height = font.size + 5 if hasattr(font, "size") else 40

    # Calculate total height of all lines
    total_height = len(lines) * line_height
    start_y = y - total_height // 2

    for i, line in enumerate(lines):
        bbox = draw.textbbox((0, 0), line, font=font)
        text_width = bbox[2] - bbox[0]
        x = (VIDEO_WIDTH - text_width) // 2
        line_y = start_y + i * line_height
        draw.text((x, line_y), line, font=font, fill=color)


def _draw_event_badges(
    draw: ImageDraw.ImageDraw,
    events_with_age: list[tuple[TrackEvent, float]],
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> None:
    """Draw event badges in a horizontal row.

    Events fade out based on their age.

    Args:
        draw: ImageDraw object to draw on.
        events_with_age: List of (event, age_seconds) tuples.
        font: Font to use for badge text.
    """
    if not events_with_age:
        return

    # Group events by type (enter on left, exit on right)
    enter_events = [(e, a) for e, a in events_with_age if e.event_type == "enter"]
    exit_events = [(e, a) for e, a in events_with_age if e.event_type == "exit"]

    # Calculate positions
    badge_spacing = 20
    current_x: float = 40

    # Draw enter events on the left
    for event, age in enter_events:
        badge_text = format_event_badge(event)
        color = get_category_color(event.category)

        # Apply fade based on age
        fade_factor = max(0, 1 - age / EVENT_FADE_SECONDS)
        faded_color = tuple(int(c * fade_factor) for c in color)

        draw.text((current_x, EVENT_ROW_Y), badge_text, font=font, fill=faded_color)

        bbox = draw.textbbox((current_x, EVENT_ROW_Y), badge_text, font=font)
        current_x = bbox[2] + badge_spacing

    # Draw exit events on the right
    if exit_events:
        # Calculate total width of exit events
        exit_texts = [format_event_badge(e) for e, _ in exit_events]
        exit_widths = [
            draw.textbbox((0, 0), t, font=font)[2] - draw.textbbox((0, 0), t, font=font)[0]
            for t in exit_texts
        ]
        total_exit_width = sum(exit_widths) + badge_spacing * (len(exit_events) - 1)

        current_x = VIDEO_WIDTH - 40 - total_exit_width

        for (event, age), badge_text in zip(exit_events, exit_texts, strict=True):
            color = get_category_color(event.category)

            # Apply fade based on age
            fade_factor = max(0, 1 - age / EVENT_FADE_SECONDS)
            faded_color = tuple(int(c * fade_factor) for c in color)

            draw.text((current_x, EVENT_ROW_Y), badge_text, font=font, fill=faded_color)

            bbox = draw.textbbox((current_x, EVENT_ROW_Y), badge_text, font=font)
            current_x = bbox[2] + badge_spacing


def _draw_progress_bar(draw: ImageDraw.ImageDraw, progress: float) -> None:
    """Draw the progress bar at the bottom of the frame.

    Args:
        draw: ImageDraw object to draw on.
        progress: Progress value from 0.0 to 1.0.
    """
    # Draw background
    draw.rectangle(
        [
            PROGRESS_BAR_X,
            PROGRESS_BAR_Y,
            PROGRESS_BAR_X + PROGRESS_BAR_WIDTH,
            PROGRESS_BAR_Y + PROGRESS_BAR_HEIGHT,
        ],
        fill=PROGRESS_BAR_BG_COLOR,
    )

    # Draw fill
    fill_width = int(PROGRESS_BAR_WIDTH * progress)
    if fill_width > 0:
        draw.rectangle(
            [
                PROGRESS_BAR_X,
                PROGRESS_BAR_Y,
                PROGRESS_BAR_X + fill_width,
                PROGRESS_BAR_Y + PROGRESS_BAR_HEIGHT,
            ],
            fill=PROGRESS_BAR_FILL_COLOR,
        )


def encode_video_with_ffmpeg(
    frames_dir: Path,
    output_path: Path,
    audio_path: Path | None = None,
) -> None:
    """Encode frames to MP4 using ffmpeg.

    Args:
        frames_dir: Directory containing numbered PNG frames.
        output_path: Path to write the output MP4 file.
        audio_path: Optional audio file to mux into the video.

    Raises:
        RuntimeError: If ffmpeg is not available or encoding fails.
    """
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise RuntimeError(
            "ffmpeg not found. Please install ffmpeg to generate videos. "
            "On macOS: brew install ffmpeg. "
            "On Ubuntu: sudo apt install ffmpeg."
        )

    # Build ffmpeg command
    # Note: All inputs must come before output options
    cmd = [
        ffmpeg_path,
        "-y",  # Overwrite output
        "-framerate",
        str(FRAME_RATE),
        "-i",
        str(frames_dir / "frame_%06d.png"),
    ]

    # Add audio input if provided (must come before output options)
    if audio_path is not None:
        cmd.extend(["-i", str(audio_path)])

    # Output options (after all inputs)
    cmd.extend(
        [
            "-c:v",
            "libx264",
            "-crf",
            str(VIDEO_CRF),
            "-pix_fmt",
            "yuv420p",  # Compatibility
        ]
    )

    # Add audio output options if audio was provided
    if audio_path is not None:
        cmd.extend(["-c:a", "aac", "-b:a", "192k", "-shortest"])

    cmd.append(str(output_path))

    logger.info("Running ffmpeg: %s", " ".join(cmd))

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr}")


def generate_visualizer(
    phrases: list[Phrase],
    bpm: float,
    total_beats: float,
    output_path: Path,
    audio_path: Path | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> Path:
    """Generate a lyrics/cues visualization video.

    Renders frames showing lyrics, section markers, track events, and a
    progress bar, then encodes them to an MP4 video.

    Args:
        phrases: List of phrases with lyrics, events, and section info.
        bpm: Tempo in beats per minute.
        total_beats: Total song duration in beats.
        output_path: Path to write the output MP4 file.
        audio_path: Optional audio file to mux into the video.
        progress_callback: Optional callback called with (frame_num, total_frames)
            for progress updates.

    Returns:
        Path to the generated video file.

    Raises:
        RuntimeError: If video generation fails.
    """
    if not phrases:
        raise ValueError("No phrases provided for visualization")

    # Calculate total duration and frames
    total_time = total_beats * 60 / bpm
    total_frames = int(total_time * FRAME_RATE)

    if total_frames == 0:
        raise ValueError("Video would have zero frames")

    logger.info(
        "Generating %d frames at %d fps for %.1f seconds",
        total_frames,
        FRAME_RATE,
        total_time,
    )

    # Load fonts
    fonts = FontSet.load()

    # Create temporary directory for frames
    with tempfile.TemporaryDirectory(prefix="alsmuse_viz_") as temp_dir:
        frames_path = Path(temp_dir)

        # Generate frames
        frame_generator = build_frame_states(phrases, bpm, total_beats)

        for frame_num, state in enumerate(frame_generator):
            img = render_frame(state, fonts)
            frame_path = frames_path / f"frame_{frame_num:06d}.png"
            img.save(frame_path)

            if progress_callback is not None:
                progress_callback(frame_num + 1, total_frames)

        # Encode video
        encode_video_with_ffmpeg(frames_path, output_path, audio_path)

    logger.info("Video generated: %s", output_path)
    return output_path
