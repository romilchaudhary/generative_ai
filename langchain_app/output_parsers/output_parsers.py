from __future__ import annotations
import json
import re
import csv
from io import StringIO
from typing import Any, Dict, List, Optional, Pattern

"""
Simple set of output parsers similar to ones you might use with LangChain.
Save as: /C:/Users/romil/OneDrive/Desktop/Romil/genai/examples/langchain_app/calculator/output_parsers/output_parsers.py
"""



class OutputParser:
    """Base parser interface."""

    def parse(self, text: str) -> Any:
        """Parse raw model output into Python structures."""
        raise NotImplementedError

    def get_format_instructions(self) -> str:
        """Optional human-readable instructions for expected format."""
        return ""


class JSONOutputParser(OutputParser):
    """Parse JSON strings into Python objects."""

    def parse(self, text: str) -> Any:
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            # Try to be forgiving: strip surrounding backticks or ```blocks```
            cleaned = text.strip().strip("` \n")
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                raise ValueError(f"Invalid JSON output: {e}") from e

    def get_format_instructions(self) -> str:
        return "Respond with valid JSON."


class KeyValueOutputParser(OutputParser):
    """Parse simple key: value style outputs into a dict.

    Accepts lines like:
      name: Alice
      score: 42
    """

    def __init__(self, pair_delimiter: str = ":"):
        self.pair_delimiter = pair_delimiter

    def parse(self, text: str) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for line in text.splitlines():
            if not line.strip():
                continue
            if self.pair_delimiter not in line:
                # skip or raise depending on policy; here we skip malformed lines
                continue
            key, val = line.split(self.pair_delimiter, 1)
            out[key.strip()] = val.strip()
        return out

    def get_format_instructions(self) -> str:
        return "Respond with one key: value pair per line."


class RegexOutputParser(OutputParser):
    """Use a regex pattern to extract named groups or whole matches.

    If pattern has named groups, returns a dict of groups for the first match.
    Otherwise returns a list of string matches.
    """

    def __init__(self, pattern: str, flags: int = 0):
        self.pattern: Pattern = re.compile(pattern, flags)

    def parse(self, text: str) -> Any:
        m = self.pattern.search(text)
        if not m:
            raise ValueError("No match found for pattern.")
        if m.groupdict():
            return m.groupdict()
        # return all full matches if no named groups
        return [match.group(0) for match in self.pattern.finditer(text)]

    def get_format_instructions(self) -> str:
        return f"Response should match regex: {self.pattern.pattern}"


class CSVOutputParser(OutputParser):
    """Parse CSV formatted text into list of rows (each row is a list of columns)."""

    def __init__(self, has_header: bool = False):
        self.has_header = has_header

    def parse(self, text: str) -> List[List[str]]:
        f = StringIO(text.strip())
        reader = csv.reader(f)
        rows = [row for row in reader if any(cell.strip() for cell in row)]
        if self.has_header and rows:
            header = rows[0]
            data = rows[1:]
            # return as list of dicts if header present
            return [dict(zip(header, row)) for row in data]  # type: ignore
        return rows

    def get_format_instructions(self) -> str:
        if self.has_header:
            return "Respond with CSV including a header row."
        return "Respond with CSV rows."


if __name__ == "__main__":
    # Simple examples showing how to use the parsers.

    json_text = '{"name": "Alice", "score": 95}'
    print("JSONOutputParser:", JSONOutputParser().parse(json_text))

    messy_json = "```\n{\n  \"a\": 1,\n  \"b\": 2\n}\n```"
    print("JSONOutputParser (messy):", JSONOutputParser().parse(messy_json))

    kv_text = "name: Bob\nrole: developer\nempty_line_above:\n\nignored line\nage: 30"
    print("KeyValueOutputParser:", KeyValueOutputParser().parse(kv_text))

    regex_text = "Result: value=123; id=xyz\nAnother line"
    parser = RegexOutputParser(r"value=(?P<value>\d+); id=(?P<id>\w+)")
    print("RegexOutputParser:", parser.parse(regex_text))

    csv_text = "name,score\nAlice,90\nBob,85\n"
    print("CSVOutputParser (with header):", CSVOutputParser(has_header=True).parse(csv_text))

    csv_simple = "a,b,c\n1,2,3\n4,5,6\n"
    print("CSVOutputParser (rows):", CSVOutputParser(has_header=False).parse(csv_simple))