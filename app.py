from server import app
from server.routes import http_routes

'''
@TODO: Improvements:
- Include hltv python lib to remove depenency with match dataset (hltv crawler)
- Implement Model Selection
'''


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
