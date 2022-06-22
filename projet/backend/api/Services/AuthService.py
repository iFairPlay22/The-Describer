from dotenv import dotenv_values


def verify_token(headers):
    auth = headers.get('Authorization')
    if auth is None:
        return False
    token = auth.split()[1]
    config = dotenv_values("./config/.env.local")
    if token != config["TOKEN"]:
        return False
    return True
