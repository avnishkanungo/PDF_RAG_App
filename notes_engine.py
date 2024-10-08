from llama_index.core.tools import FunctionTool
import os

note_file = os.getcwd() + "/data/notes.txt"


def save_note(note):
    if not os.path.exists(note_file):
        open(note_file, "w")

    with open(note_file, "a") as f:
        f.writelines([note + "\n\n"])

    return "note saved"


note_engine = FunctionTool.from_defaults(
    fn=save_note,
    name="note_saver",
    description="this tool can save a text based note to a file for the user, when there is a prompt asking to take a note. When this tool is used do not use the mcq_test_generation action",
)