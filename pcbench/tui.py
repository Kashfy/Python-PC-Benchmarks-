"""Raw-terminal widgets: arrow-key lists, checkboxes, and a text field.

The guided menu should be driven the way an OS installer is — move with the
arrow keys, toggle with space, Enter to accept — not by typing a number and
pressing return. That needs the terminal in raw mode and ANSI drawing, both
of which the standard library can do on every platform pcbench targets, so
this module does it by hand rather than taking a dependency: termios and tty
on Unix, msvcrt on Windows.

When the terminal cannot do that — output is piped, TERM is dumb, or this is
a test — every widget falls back to the typed prompt it replaced. The same
flow then still works over a pipe, which is what keeps the menu scriptable
and testable.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import sys

try:                                        # POSIX
    import select as _select
    import termios
    import tty
    _HAVE_TERMIOS = True
except ImportError:                         # Windows
    _HAVE_TERMIOS = False

try:                                        # Windows
    import msvcrt
    _HAVE_MSVCRT = True
except ImportError:
    _HAVE_MSVCRT = False


class Back(Exception):
    """The user asked for the previous screen."""


class Quit(Exception):
    """The user asked to leave without running anything."""


CSI = "\x1b["
_ALT_ON, _ALT_OFF = f"{CSI}?1049h", f"{CSI}?1049l"
_CURSOR_OFF, _CURSOR_ON = f"{CSI}?25l", f"{CSI}?25h"
_CLEAR = f"{CSI}2J{CSI}H"
_REVERSE, _DIM, _BOLD, _RESET = f"{CSI}7m", f"{CSI}2m", f"{CSI}1m", f"{CSI}0m"

#: Widest the frame is drawn, however wide the window is: a list that
#: runs the full width of a maximised terminal is harder to scan, not
#: easier.
MAX_WIDTH = 100
MIN_WIDTH = 50

#: Key hints, widest first. A narrow window gets the shorter wording rather
#: than a truncated line that hides the key for leaving.
_FOOTER_SELECT = (
    "[up/down] move   [enter] select   [esc] back   [q] quit",
    "[enter] select   [esc] back   [q] quit",
    "[enter] ok  [esc] back  [q] quit",
)
_FOOTER_MULTI = (
    "[up/down] move   [space] toggle   [a] all/none   "
    "[enter] accept   [esc] back   [q] quit",
    "[space] toggle   [a] all/none   [enter] accept   [esc] back   [q] quit",
    "[space] toggle   [enter] accept   [esc] back   [q] quit",
    "[space] pick  [enter] ok  [esc] back  [q] quit",
)
_FOOTER_TEXT = (
    "[type] answer   [enter] accept   [esc] back",
    "[enter] accept   [esc] back",
)

#: Typed-prompt footers, for the fallback path where there are no keys to
#: press, only words to type.
_TYPED_SELECT = "  [number] choose    [b] back    [q] quit"
_TYPED_MULTI = "  [numbers] choose    [b] back    [q] quit"
_TYPED_TEXT = "  [value] answer    [b] back    [q] quit"

_BACK_WORDS = {"b", "back"}
_QUIT_WORDS = {"q", "quit", "exit"}


# --------------------------------------------------------------------------- #
# Capability detection
# --------------------------------------------------------------------------- #
def _enable_windows_vt() -> bool:
    """Turn on ANSI processing, which Windows 10+ has but leaves off."""
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)          # STD_OUTPUT_HANDLE
        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        # ENABLE_VIRTUAL_TERMINAL_PROCESSING
        return bool(kernel32.SetConsoleMode(handle, mode.value | 0x0004))
    except Exception:
        return False


def supported() -> bool:
    """Whether this terminal can be driven key by key, not line by line."""
    if os.environ.get("PCBENCH_NO_TUI"):
        return False
    try:
        if not (sys.stdin.isatty() and sys.stdout.isatty()):
            return False
    except (AttributeError, ValueError):
        return False
    if os.name == "nt":
        return _HAVE_MSVCRT and _enable_windows_vt()
    if os.environ.get("TERM", "") in ("", "dumb"):
        return False
    return _HAVE_TERMIOS


# --------------------------------------------------------------------------- #
# Key reading
# --------------------------------------------------------------------------- #
#: What the tail of an ANSI escape sequence means. Terminals disagree about
#: the prefix (``ESC [`` or ``ESC O``) but not about these.
_SEQUENCES = {
    "A": "UP", "B": "DOWN", "C": "RIGHT", "D": "LEFT",
    "H": "HOME", "F": "END", "1~": "HOME", "4~": "END",
    "5~": "PGUP", "6~": "PGDN", "3~": "DELETE",
}


def _read_posix() -> str:
    fd = sys.stdin.fileno()
    first = os.read(fd, 1).decode("utf-8", "ignore")
    if first != "\x1b":
        return first
    # An escape byte starts a sequence, or is the Esc key on its own. Only a
    # timeout tells them apart, because nothing follows a bare Esc.
    if not _select.select([sys.stdin], [], [], 0.05)[0]:
        return "ESC"
    second = os.read(fd, 1).decode("utf-8", "ignore")
    if second not in ("[", "O"):
        return "ESC"
    tail = ""
    while len(tail) < 8:
        if not _select.select([sys.stdin], [], [], 0.05)[0]:
            break
        char = os.read(fd, 1).decode("utf-8", "ignore")
        tail += char
        if char.isalpha() or char == "~":
            break
    return _SEQUENCES.get(tail, "")


def _read_windows() -> str:
    char = msvcrt.getwch()
    if char in ("\x00", "\xe0"):
        return {"H": "UP", "P": "DOWN", "K": "LEFT", "M": "RIGHT",
                "G": "HOME", "O": "END", "I": "PGUP", "Q": "PGDN",
                "S": "DELETE"}.get(msvcrt.getwch(), "")
    return char


def read_key() -> str:
    """One keypress, as a name (``UP``, ``ENTER``) or the character itself."""
    char = _read_windows() if os.name == "nt" else _read_posix()
    if char in ("\r", "\n"):
        return "ENTER"
    if char in ("\x7f", "\b"):
        return "BACKSPACE"
    if char == "\t":
        return "TAB"
    if char == "":
        return ""
    return char


# --------------------------------------------------------------------------- #
# Screen
# --------------------------------------------------------------------------- #
@contextlib.contextmanager
def screen():
    """Own the terminal for the duration: alternate buffer, raw mode, no cursor.

    Everything is restored on the way out however the block ends, including
    an exception, because leaving a terminal in raw mode makes the user's
    shell unusable.
    """
    if not supported():
        yield False
        return
    fd, saved = None, None
    try:
        fd = sys.stdin.fileno()
        saved = termios.tcgetattr(fd) if _HAVE_TERMIOS else None
    except Exception:
        fd, saved = None, None
    sys.stdout.write(_ALT_ON + _CURSOR_OFF)
    sys.stdout.flush()
    try:
        if saved is not None:
            tty.setraw(fd)
        yield True
    finally:
        if saved is not None:
            termios.tcsetattr(fd, termios.TCSADRAIN, saved)
        sys.stdout.write(_CURSOR_ON + _ALT_OFF)
        sys.stdout.flush()


def _size() -> tuple[int, int]:
    columns, rows = shutil.get_terminal_size((80, 24))
    return max(MIN_WIDTH, min(columns, MAX_WIDTH)), max(12, rows)


def fit(text: str, width: int) -> str:
    return text if len(text) <= width else text[:max(0, width - 3)] + "..."


def _paint(out, lines) -> None:
    # Raw mode turns off the newline translation, so every break has to
    # carry its own carriage return or the text walks off to the right.
    out.write(_CLEAR + "\r\n".join(lines) + "\r\n")
    out.flush()


def _widest_that_fits(candidates, width: int) -> str:
    for text in candidates:
        if len(text) <= width - 4:
            return text
    return fit(candidates[-1], width - 4)


def _frame_top(title: str, width: int) -> list[str]:
    return ["=" * width, "  " + fit(title, width - 4), "=" * width]


def _window(count: int, cursor: int, room: int) -> tuple[int, int]:
    """The slice of a long list to show, keeping the cursor inside it."""
    if count <= room:
        return 0, count
    start = max(0, min(cursor - room // 2, count - room))
    return start, start + room


def _row(option, width: int, label_width: int, mark: str) -> str:
    label = option[0]
    detail = option[1] if len(option) > 1 else ""
    text = f"{mark}{label:<{label_width}}"
    if detail:
        text = f"{text}  {detail}"
    return fit(text.rstrip(), width - 6)


def _paint_list(out, title, question, options, body, note, cursor, marked,
                error, footer) -> None:
    width, height = _size()
    lines = _frame_top(title, width)
    lines += ["", f"  {_BOLD}{question}{_RESET}", ""]
    if body:
        lines += list(body) + [""]

    tail = []
    if note:
        tail += ["", f"  {_DIM}{note}{_RESET}"]
    if error:
        tail += ["", f"  !  {error}"]
    hint = _widest_that_fits(footer, width)
    tail += ["", f"  {_DIM}{hint}{_RESET}"]

    room = max(3, height - len(lines) - len(tail) - 2)
    start, end = _window(len(options), cursor, room)
    labels = [o[0] for o in options]
    label_width = min(max((len(x) for x in labels), default=0),
                      max(8, width // 2))

    if start > 0:
        lines.append(f"      {_DIM}^ {start} more{_RESET}")
    for index in range(start, end):
        mark = ""
        if marked is not None:
            mark = "[x] " if index in marked else "[ ] "
        text = _row(options[index], width, label_width, mark)
        if index == cursor:
            lines.append(f"   {_REVERSE} > {text.ljust(width - 6)} {_RESET}")
        else:
            lines.append(f"      {text}")
    if end < len(options):
        lines.append(f"      {_DIM}v {len(options) - end} more{_RESET}")

    _paint(out, lines + tail)


# --------------------------------------------------------------------------- #
# Widgets — key-driven
# --------------------------------------------------------------------------- #
def _move(key: str, cursor: int, count: int) -> int | None:
    """The new cursor for a navigation key, or None if it was not one."""
    if key in ("UP", "k"):
        return (cursor - 1) % count
    if key in ("DOWN", "j"):
        return (cursor + 1) % count
    if key == "HOME":
        return 0
    if key == "END":
        return count - 1
    if key == "PGUP":
        return max(0, cursor - 10)
    if key == "PGDN":
        return min(count - 1, cursor + 10)
    if key.isdigit() and key != "0" and int(key) <= count:
        return int(key) - 1
    return None


def _select_keys(title, question, options, body, note, read, out) -> int:
    cursor = 0
    while True:
        _paint_list(out, title, question, options, body, note, cursor, None,
                    "", _FOOTER_SELECT)
        key = read()
        moved = _move(key, cursor, len(options))
        if moved is not None:
            cursor = moved
        elif key in ("ENTER", "RIGHT", " "):
            return cursor
        elif key in ("ESC", "LEFT"):
            raise Back
        elif key in ("q", "\x03"):
            raise Quit


def _multi_keys(title, question, options, chosen, note, allow_empty,
                read, out) -> list[int]:
    marked, cursor, error = set(chosen), 0, ""
    while True:
        _paint_list(out, title, question, options, (), note, cursor, marked,
                    error, _FOOTER_MULTI)
        key = read()
        moved = _move(key, cursor, len(options))
        if moved is not None:
            cursor = moved
        elif key == " ":
            marked.symmetric_difference_update({cursor})
        elif key == "a":
            marked = (set() if len(marked) == len(options)
                      else set(range(len(options))))
        elif key == "ENTER":
            if marked or allow_empty:
                return sorted(marked)
            error = "choose at least one, or press esc to go back"
        elif key in ("ESC", "LEFT"):
            raise Back
        elif key in ("q", "\x03"):
            raise Quit


def _text_keys(title, question, label, default, body, validate,
               read, out) -> str:
    typed, error = "", ""
    while True:
        shown = typed if typed else f"{_DIM}{default}{_RESET}"
        field = [f"  {label}: {shown}{_REVERSE} {_RESET}"]
        _paint_list(out, title, question, [], list(body) + [""] + field, "",
                    0, None, error, _FOOTER_TEXT)
        key = read()
        if key == "ENTER":
            value = typed or default
            if validate is None:
                return value
            try:
                validate(value)
                return value
            except ValueError as e:
                error, typed = str(e), ""
        elif key == "BACKSPACE":
            typed = typed[:-1]
        elif key == "\x15":                      # Ctrl-U
            typed = ""
        elif key == "ESC":
            raise Back
        elif key == "\x03":
            raise Quit
        elif len(key) == 1 and key.isprintable():
            typed += key


# --------------------------------------------------------------------------- #
# Widgets — typed fallback
# --------------------------------------------------------------------------- #
def _typed_header(title: str) -> None:
    print()
    print("=" * 74)
    print("  " + fit(title, 70))
    print("=" * 74)


def _typed_draw(title, question, lines, error, footer) -> None:
    _typed_header(title)
    print(f"\n  {question}\n")
    for line in lines:
        print(line)
    if error:
        print(f"\n  !  {error}")
    print(f"\n{footer}")


def _typed_input(label: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    try:
        answer = input(f"\n  {label}{suffix}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        raise Quit
    lowered = answer.lower()
    if lowered in _QUIT_WORDS:
        raise Quit
    if lowered in _BACK_WORDS:
        raise Back
    return answer or default


def _typed_options(options, marked=None) -> list[str]:
    lines = []
    for number, option in enumerate(options, 1):
        mark = ""
        if marked is not None:
            mark = "[x] " if number - 1 in marked else "[ ] "
        detail = option[1] if len(option) > 1 else ""
        lines.append(f"    {number:>2}. {mark}{option[0]}")
        if detail:
            lines.append("        " + fit(detail, 64))
    return lines


def _select_typed(title, question, options, body, note) -> int:
    error = ""
    while True:
        lines = list(body) + ([""] if body else []) + _typed_options(options)
        if note:
            lines += ["", f"  {note}"]
        _typed_draw(title, question, lines, error, _TYPED_SELECT)
        answer = _typed_input("Choice")
        if answer.isdigit() and 1 <= int(answer) <= len(options):
            return int(answer) - 1
        error = f"{answer!r} is not one of 1-{len(options)}"


def parse_selection(answer: str, count: int, names=None) -> list[int]:
    """Turn ``'1,4-6'``, ``'all'``, or ``'cpu_int,disk'`` into indexes."""
    lowered = answer.strip().lower()
    if lowered in ("all", "*"):
        return list(range(count))
    if lowered in ("none", "-"):
        return []
    chosen: list[int] = []
    for token in lowered.replace(" ", ",").split(","):
        token = token.strip()
        if not token:
            continue
        parts = token.split("-", 1)
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            low, high = int(parts[0]), int(parts[1])
            if not 1 <= low <= high <= count:
                raise ValueError(f"{token!r} is outside 1-{count}")
            chosen += list(range(low - 1, high))
        elif token.isdigit():
            number = int(token)
            if not 1 <= number <= count:
                raise ValueError(f"{token!r} is outside 1-{count}")
            chosen.append(number - 1)
        elif names and token in names:
            chosen.append(names.index(token))
        else:
            raise ValueError(f"{token!r} is not one of the choices")
    seen, ordered = set(), []
    for index in chosen:
        if index not in seen:
            seen.add(index)
            ordered.append(index)
    return ordered


def _multi_typed(title, question, options, names, default, allow_empty,
                 note) -> list[int]:
    hint = ("Numbers (1,4), ranges (1-6), 'all', or 'none'"
            + (", or names" if names else ""))
    error = ""
    while True:
        lines = _typed_options(options) + ["", f"  {note or hint}"]
        _typed_draw(title, question, lines, error, _TYPED_MULTI)
        answer = _typed_input("Choice(s)", default)
        try:
            chosen = parse_selection(answer, len(options), names)
        except ValueError as e:
            error = str(e)
            continue
        if not chosen and not allow_empty:
            error = "choose at least one"
            continue
        return chosen


def _text_typed(title, question, label, default, body, validate) -> str:
    error = ""
    while True:
        _typed_draw(title, question, list(body), error, _TYPED_TEXT)
        answer = _typed_input(label, default)
        if validate is None:
            return answer
        try:
            validate(answer)
            return answer
        except ValueError as e:
            error = str(e)


# --------------------------------------------------------------------------- #
# Public widgets: keys when the terminal allows it, typing when it does not
# --------------------------------------------------------------------------- #
def select(title, question, options, body=(), note="", read=None,
           out=None) -> int:
    """Pick one of ``options``. Returns its index."""
    if read is None and not supported():
        return _select_typed(title, question, options, body, note)
    return _select_keys(title, question, options, body, note,
                        read or read_key, out or sys.stdout)


def multiselect(title, question, options, names=None, default="none",
                allow_empty=True, note="", read=None, out=None) -> list[int]:
    """Toggle any number of ``options``. Returns the chosen indexes."""
    if read is None and not supported():
        return _multi_typed(title, question, options, names, default,
                            allow_empty, note)
    # The typed path takes a default answer; the key-driven one starts with
    # those same entries already ticked.
    chosen = parse_selection(default or "none", len(options), names)
    return _multi_keys(title, question, options, chosen, note, allow_empty,
                       read or read_key, out or sys.stdout)


def text(title, question, label, default="", body=(), validate=None,
         read=None, out=None) -> str:
    """Read a line of text, re-asking until ``validate`` accepts it."""
    if read is None and not supported():
        return _text_typed(title, question, label, default, body, validate)
    return _text_keys(title, question, label, default, body, validate,
                      read or read_key, out or sys.stdout)
