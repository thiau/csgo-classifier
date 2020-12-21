import pickle
import json
from server import app, CORS, jsonify, request

CORS(app)
resource_path = "server/resources"


def load_team_models(team_id):
    classifier = pickle.load(
        open(f"{resource_path}/models/classifiers/{team_id}.pickle", 'rb'))
    encoder = pickle.load(
        open(f"{resource_path}/models/encoders/{team_id}.pickle", 'rb'))
    return classifier, encoder


def get_match_prediction(team, oposite_team, game_map):
    classifier, encoder = load_team_models(team.get("id"))
    ds = [[oposite_team.get("name"), game_map]]
    X = encoder.transform(ds).toarray()
    y_pred = classifier.predict(X)
    return y_pred


@app.route("/predict", methods=["POST"])
def match():
    params = request.form

    game_map = params.get("game_map")

    team = {"name": params.get("team_name"), "id": params.get("team_id")}

    oposite_team = {
        "name": params.get("oposite_team_name"),
        "id": params.get("oposite_team_id")
    }

    prediction = get_match_prediction(team, oposite_team, game_map)
    victory = True if prediction[0] else False
    return jsonify(victory=victory)
