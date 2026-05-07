from langchain_community.document_loaders import WebBaseLoader


url = "https://www.google.com/search?gs_ssp=eJzjYtfP1TfItqhSYDRgdGDw4k4sKMhJVSguyS9KBQBZIQdw&q=apple+store&oq=Apple&gs_lcrp=EgZjaHJvbWUqEAgBEC4YxwEYsQMY0QMYgAQyDggAEEUYJxg7GIAEGIoFMhAIARAuGMcBGLEDGNEDGIAEMggIAhBFGCcYOzIGCAMQRRg7MgYIBBBFGDwyBggFEEUYPDIGCAYQRRg8MgYIBxBFGDzSAQgyOTI5ajBqN6gCALACAA&sourceid=chrome&ie=UTF-8"

data = WebBaseLoader(url)


docs = data.load()
print(docs[0].page_content)






























