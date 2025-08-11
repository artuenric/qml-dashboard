# backend/main.py
from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

app = FastAPI()

# Permitir requisições do frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ExecRequest(BaseModel):
    algorithm: str
    dataset: str
    cloud: str
    token: str

@app.post("/execute")
def run_notebook(req: ExecRequest):
    if req.algorithm == "QSVC":
        return execute_notebook("Inference.ipynb")
    else:
        return {"status": "erro", "mensagem": "Algoritmo não implementado"}

def execute_notebook(notebook_path):
    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')

    try:
        ep.preprocess(nb, {'metadata': {'path': './'}})
        # Suponha que o notebook escreva resultado final no arquivo result.txt
        with open("result.txt") as r:
            result = r.read()
        return {"status": "ok", "resultado": result}
    except Exception as e:
        return {"status": "erro", "mensagem": str(e)}
