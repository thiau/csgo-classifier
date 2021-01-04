import os
import json
import pickle
import pandas as pd
from server.helpers.classifier import Classifier

# Base folders path
resources_path = "server/resources"
models_path = "server/resources/models"

# Folders path
datasets_path = f"{resources_path}/datasets"
encoders_path = f"{models_path}/encoders"
classifiers_path = f"{models_path}/classifiers"
team_data_path = f"{models_path}/team_data"


def create_folders():
    os.makedirs(os.path.dirname(f"{datasets_path}/"), exist_ok=True)
    os.makedirs(os.path.dirname(f"{encoders_path}/"), exist_ok=True)
    os.makedirs(os.path.dirname(f"{classifiers_path}/"), exist_ok=True)
    os.makedirs(os.path.dirname(f"{team_data_path}/"), exist_ok=True)

def load_resources():
    data = None
    with open(f"{datasets_path}/matches.json") as json_file:
        data = json.load(json_file)

    teams = None
    with open(f"{datasets_path}/ranking.json") as json_file:
        teams = json.load(json_file)

    return data, teams


def generate_csv_from_dict(data: dict, file_name: str):
    dataframe = pd.DataFrame(data)
    dataframe.to_csv(
        f"{team_data_path}/{file_name}.csv",
        sep=";",
        index=False)


def get_team_data(team_id, data):
    team_data = list()
    for match in data:
        if match["team1"]["id"] == team_id or match["team2"]["id"] == team_id:
            if match["result"]["team1"] != match["result"]["team2"]:
                current_position = str()
                oposite_position = str()

                if match["team1"]["id"] == team_id:
                    current_position = "team1"
                    oposite_position = "team2"
                else:
                    current_position = "team2"
                    oposite_position = "team1"

                win = 1 if match["result"][current_position] > match["result"][
                    oposite_position] else 0

                team_data.append({
                    "oposite_team":
                    match[oposite_position]["name"],
                    "map":
                    match["map"],
                    "win":
                    win
                })

    generate_csv_from_dict(team_data, team_id)
    return team_data


def generate_models():
    create_folders()
    data, teams = load_resources()

    for team in teams:
        print("\n")
        team_id = team["team"]["id"]
        team_name = team["team"]["name"]

        team_data = get_team_data(team_id, data)
        print("Creating model for \x1b[1;31m{}\x1b[0m".format(team_name))

        ds = pd.DataFrame(team_data)

        print("Team Data Size: \x1b[1;31m{}\x1b[0m".format(len(team_data)))

        classifier = Classifier(
            data=ds, column_order=["oposite_team", "map", "win"])
        classifier.train()

        classifier_model = classifier.get_classifier_model()
        encoder_model = classifier.get_encoder_model()

        file_name = f"{team_id}.pickle"
        pickle.dump(classifier_model,
                    open(f"{models_path}/classifiers/{file_name}", 'wb'))
        pickle.dump(encoder_model,
                    open(f"{models_path}/encoders/{file_name}", 'wb'))
