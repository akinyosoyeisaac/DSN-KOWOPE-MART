import os

def check_data() -> str:
    if not os.path.exists("data\Train.csv"):
        output = "The file does not exist visit either of the following link to download the data 'https://zindi.africa/hackathons/dsn-ai-bootcamp-qualification-hackathon/data' or 'https://drive.google.com/drive/u/0/mobile/folders/1yi3Sv14HzNPo4CHoA32if7kXDmq5-DF9/1AL4d22aHkx1rnIyHHytz-Sg0YgXWxUzf?sort=13&amp;direction=a'"
        raise Exception(output)
    else:
        output = "file has been download"
        print(output)
        return output