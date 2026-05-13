from pathlib import Path

class CustomInput():

    def __init__(self) -> None:
       ... 

    def custompath(self):

        file_path = Path("./custom_input_file.txt") #just to make sure code doesnt break when run or output is changed while running program
        file_path.touch()

        with file_path.open("r") as file:
            content = file.read()
            if content == "":
                raise ValueError("no content")
            elif not Path(content.strip()).exists():
                raise ValueError(f"invalid path: {content.strip()}")
            else:
                return Path(content.strip())
