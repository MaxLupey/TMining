import openai

api_key = "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"

openai.api_key = api_key

engines = openai.Engine.list()
print(engines)
for engine in engines.data:
    print("Model:", engine.name)
    print("Description:", engine.description)
    print("Version:", engine.version)
    print()
