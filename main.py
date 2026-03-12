from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI(title="Мой простой сайт")

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello, {name}!"}

@app.get("/html", response_class=HTMLResponse)
async def get_html():
    html_content = """
    <!DOCTYPE html>
    <html>
        <head>
            <title>Простая страница</title>
        </head>
        <body>
            <h1>Привет, это HTML страница!</h1>
            <p>FastAPI умеет отдавать не только JSON.</p>
        </body>
    </html>
    """
    return html_content