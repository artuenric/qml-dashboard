# Quantum Machine Learning Site

Este projeto é uma plataforma web para execução e experimentação de algoritmos de Machine Learning Quântico, integrando um backend em FastAPI (Python) e um frontend em React + TypeScript + Vite.

---

## Descrição

- **Frontend:** Interface web moderna para seleção de datasets, algoritmos e ambiente de execução (local/cloud), envio de parâmetros e visualização dos resultados.
- **Backend:** API em FastAPI que executa notebooks Jupyter para inferência com algoritmos quânticos (ex: QSVC), processando os dados e retornando os resultados para o frontend.

---

## Rodando com Docker

A forma mais simples de rodar todo o ambiente (frontend e backend) é usando Docker.

1. **Build e Início dos Contêineres:**
   ```bash
   docker compose up --build
   ```
   *Na primeira vez, o build pode demorar alguns minutos.*

2. **Acesse as aplicações:**
   - **Frontend:** [http://localhost:5173](http://localhost:5173)
   - **Backend (Swagger UI):** [http://localhost:8000/docs](http://localhost:8000/docs)

3. **Para parar os contêineres:**
   Pressione `Ctrl + C` no terminal onde o compose está rodando, ou execute em outro terminal:
   ```bash
   docker compose down
   ```
---

## Desenvolvimento Local (Sem Docker)

### Backend (FastAPI + Python)

### Instalação e Execução

1. **(Opcional) Remova o ambiente virtual antigo:**
   ```bash
   cd backend
   rm -rf venv
   ```
2. **Crie e ative um novo ambiente virtual:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. **Instale as dependências:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Rode o backend:**
   ```bash
   uvicorn main:app --reload
   ```
5. **Acesse a documentação interativa:**
   - [http://localhost:8000/docs](http://localhost:8000/docs)

### Testando o endpoint principal

- Endpoint: `POST /execute`
- Exemplo de payload válido:
  ```json
  {
    "algorithm": "QSVC",
    "dataset": "csv1",
    "cloud": "IBM",
    "token": "qualquer_coisa"
  }
  ```
- Você pode testar pelo Swagger UI ou via cURL:
  ```bash
  curl -X POST "http://localhost:8000/execute" \
    -H "Content-Type: application/json" \
    -d '{"algorithm": "QSVC", "dataset": "csv1", "cloud": "IBM", "token": "qualquer_coisa"}'
  ```

---

## Frontend (React + Vite)

### Instalação e Execução

1. **Na raiz do projeto:**
   ```bash
   npm install
   ```
2. **Rode o servidor de desenvolvimento:**
   ```bash
   npm run dev
   ```
3. **Acesse no navegador:**
   - [http://localhost:5173](http://localhost:5173) (ou endereço exibido no terminal)

### Fluxo de uso
- Preencha os campos de Dataset, Algoritmo, Ambiente, Cloud Provider e Token (se necessário).
- Clique em "Executar" para enviar a requisição ao backend e visualizar o resultado na tela.

---

## Observações
- O backend atualmente implementa apenas o algoritmo "QSVC" no endpoint `/execute`.
- Certifique-se de que o backend esteja rodando antes de usar o frontend.
- Para reprodutibilidade, mantenha o `requirements.txt` sempre atualizado com as dependências instaladas.

---

## Estrutura do Projeto

```
qml-site/
├── backend/           # Backend FastAPI + Notebooks + dados
│   ├── main.py
│   ├── requirements.txt
│   └── ...
├── src/               # Frontend React + TypeScript
│   └── ...
├── public/
├── package.json
├── README.md
└── ...
```

---

## Contato
Dúvidas ou sugestões: abra uma issue ou entre em contato com o mantenedor do repositório.
