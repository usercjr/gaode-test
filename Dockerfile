# Node 网关镜像：docker build -t wanderai-gateway .
FROM node:22-alpine

WORKDIR /app
ENV NODE_ENV=production

COPY package.json package-lock.json ./
RUN npm ci --omit=dev

# 只拷网关运行需要的文件（页面、前端脚本、网关代码）
COPY server.js ./
COPY gateway/ gateway/
COPY src/ src/
COPY ai-chat.html output-map.html rag-search.html admin.html ./

USER node
EXPOSE 3000
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
  CMD wget -qO- http://127.0.0.1:3000/ai-chat.html >/dev/null || exit 1
CMD ["node", "server.js"]
