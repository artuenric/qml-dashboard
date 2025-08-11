# --- Estágio 1: Build da Aplicação React ---
FROM node:20-alpine AS build

# Define o diretório de trabalho
WORKDIR /app

# Copia os arquivos de definição de pacotes
COPY package.json package-lock.json ./

# Instala as dependências
RUN npm install

# Copia o restante do código do frontend
COPY . .

# Argumento que será passado pelo pipeline de CI/CD
ARG VITE_API_URL

# Builda a aplicação para produção, passando a variável de ambiente
RUN VITE_API_URL=${VITE_API_URL} npm run build

# --- Estágio 2: Servidor de Produção (Nginx) ---
FROM nginx:stable-alpine

# Copia os arquivos buildados do estágio anterior para o diretório do Nginx
COPY --from=build /app/dist /usr/share/nginx/html

# Expõe a porta 80 (padrão do Nginx)
EXPOSE 80

# Comando padrão do Nginx para iniciar o servidor
CMD ["nginx", "-g", "daemon off;"] 