from client.infercode_client import InferCodeClient


infercode = InferCodeClient()
vectors = infercode.encode(["for int i = 0", "for int i = 0"])

print(vectors )
# vectors = infercode.encode(["for int i = 0"])


# print(vectors )