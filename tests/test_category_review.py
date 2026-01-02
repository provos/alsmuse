"""Tests for category_review module."""

from __future__ import annotations

from unittest.mock import patch

from alsmuse.category_review import (
    prompt_category_review,
    review_categories_interactive,
)


class TestReviewCategoriesInteractive:
    """Tests for review_categories_interactive function."""

    def test_returns_empty_dict_when_done_immediately(self) -> None:
        """Returns empty dict when user chooses Done immediately."""
        track_names = ["Drums", "Bass", "Lead"]
        current_categories = {"Drums": "drums", "Bass": "bass", "Lead": "lead"}
        available_categories = ["drums", "bass", "vocals", "lead", "other"]

        with patch("alsmuse.category_review.Prompt.ask") as mock_ask:
            # User presses Enter immediately (empty string = done)
            mock_ask.return_value = ""

            result = review_categories_interactive(
                track_names, current_categories, available_categories
            )

        assert result == {}

    def test_returns_override_when_track_reassigned(self) -> None:
        """Returns override dict when user reassigns a track."""
        track_names = ["Drums", "Bass"]
        current_categories = {"Drums": "other", "Bass": "bass"}
        available_categories = ["drums", "bass", "other"]

        with patch("alsmuse.category_review.Prompt.ask") as mock_ask:
            # Mock the sequence of user interactions:
            # 1. Select category #2 (other) to review
            # 2. Select track #1 (Drums)
            # 3. Select category #1 (drums) as new category
            # 4. Press Enter to go back to categories
            # 5. Press Enter to finish (done)
            mock_ask.side_effect = [
                "2",  # Select "other" category (index 2 in [bass, other])
                "1",  # Select "Drums" track
                "1",  # Select "drums" as new category
                "",  # Back to tracks
                "",  # Done with categories
            ]

            result = review_categories_interactive(
                track_names, current_categories, available_categories
            )

        assert result == {"Drums": "drums"}

    def test_back_to_categories_does_not_add_override(self) -> None:
        """Selecting back doesn't add any overrides."""
        track_names = ["Drums"]
        current_categories = {"Drums": "drums"}
        available_categories = ["drums", "other"]

        with patch("alsmuse.category_review.Prompt.ask") as mock_ask:
            # User selects category, then goes back, then done
            mock_ask.side_effect = [
                "1",  # Select "drums" category
                "",  # Back to categories (no track selected)
                "",  # Done
            ]

            result = review_categories_interactive(
                track_names, current_categories, available_categories
            )

        assert result == {}


class TestPromptCategoryReview:
    """Tests for prompt_category_review function."""

    def test_returns_none_when_no_tty(self) -> None:
        """Returns None when not running in TTY."""
        track_names = ["Drums"]
        current_categories = {"Drums": "drums"}
        available_categories = ["drums", "other"]

        with patch("alsmuse.category_review.sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = False

            result = prompt_category_review(track_names, current_categories, available_categories)

        assert result is None

    def test_returns_none_when_user_declines_review(self) -> None:
        """Returns None when user declines to review."""
        track_names = ["Drums"]
        current_categories = {"Drums": "drums"}
        available_categories = ["drums", "other"]

        with patch("alsmuse.category_review.sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = True

            with patch("alsmuse.category_review.questionary") as mock_questionary:
                mock_questionary.confirm.return_value.ask.return_value = False

                with patch("builtins.print"):
                    result = prompt_category_review(
                        track_names, current_categories, available_categories
                    )

        assert result is None

    def test_calls_review_when_user_accepts(self) -> None:
        """Calls review_categories_interactive when user accepts."""
        track_names = ["Drums"]
        current_categories = {"Drums": "drums"}
        available_categories = ["drums", "other"]

        with patch("alsmuse.category_review.sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = True

            with patch("alsmuse.category_review.questionary") as mock_questionary:
                # User accepts review
                mock_questionary.confirm.return_value.ask.return_value = True

                with patch("alsmuse.category_review.Prompt.ask") as mock_ask:
                    # User immediately presses Enter to finish
                    mock_ask.return_value = ""

                    with patch("builtins.print"):
                        result = prompt_category_review(
                            track_names, current_categories, available_categories
                        )

        # Should return empty dict (no overrides made)
        assert result == {}
