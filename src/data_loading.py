import os

def check_data() -> str:
    if not os.path.exists("data\Train.csv"):
        output = "The file does not exist visit either of the following link to download the data 'https://zindi.africa/hackathons/dsn-ai-bootcamp-qualification-hackathon/data' or 'https://drive.google.com/drive/folders/1AL4d22aHkx1rnIyHHytz-Sg0YgXWxUzf'"
        raise Exception(output)
    else:
        output = "file has been download"
        print(output)
        return output