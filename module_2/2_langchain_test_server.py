from langserve import RemoteRunnable

remote_chain = RemoteRunnable("http://localhost:8000/note/")
print(remote_chain.invoke({"topic": "Attention mechanism"}))