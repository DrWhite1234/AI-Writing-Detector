import transformers
from peft import AutoPeftModelForSequenceClassification

transformers.logging.set_verbosity_error()

model = AutoPeftModelForSequenceClassification.from_pretrained(
    "gouwsxander/slop-detector-bert"
)

classifier = transformers.pipeline(
    "text-classification",
    model=model,
    tokenizer="gouwsxander/slop-detector-bert",
)

classifier.model.config.id2label = {0: "Human", 1: "AI"}

while True:

    text = input(">> ")

    results = classifier(text)
    assert isinstance(results, list)
    assert isinstance(results[0], dict)

    label = results[0]["label"]
    score = results[0]["score"]

    print(f"\nResult: {label} ({100 * score:.2f}%)\n")

    