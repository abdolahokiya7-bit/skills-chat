# Website (chat page + skills API). Render uses this file.
FROM node:22-slim
WORKDIR /app
COPY server.js index.html skills.json ./
ENV NODE_ENV=production
EXPOSE 3000
CMD ["node", "server.js"]
