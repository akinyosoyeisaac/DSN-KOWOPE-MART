import os
from log import get_logger
import yaml
import typer
app = typer.Typer()


def configuration(path:str)->dict:
    with open(path, "r") as file:
        config_file = yaml.safe_load(file)
    return config_file


@app.command()
def check_data(path: str):
    config = configuration(path)
    log = get_logger("DATA LOADING", config["loglevel"])
    log.msg("Checking if the train file has been download")
    if not os.path.exists(config["data_loading"]["train_path"]):
        output = "The file does not exist visit either of the following link to download the data 'https://zindi.africa/hackathons/dsn-ai-bootcamp-qualification-hackathon/data' or 'https://drive.google.com/drive/folders/1AL4d22aHkx1rnIyHHytz-Sg0YgXWxUzf'"
        log.msg("Please check the exception error message to download the train and test files")
        raise Exception(output)
    else:
        output = "file has been download"
        log.msg(output)
    
if __name__ == "__main__":
    app()