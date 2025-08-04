from mangum import Mangum
from app.main import app

handler = Mangum(app)

def handler_func(event, context):
    return handler(event, context)
