import openai

api_key = "YOUR_API_KEY"

openai.api_key = api_key

engines = openai.Engine.list()
print(engines)
for engine in engines.data:
    print("Model:", engine.name)
    print("Description:", engine.description)
    print("Version:", engine.version)
    print()
