from fastapi import FastAPI, Body
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = FastAPI()

model_name = "habanoz/distilbert-base-multilingual-cased-5000-raw-lr2_5-3ep"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name).to("cuda")


class Inputs(BaseModel):
    items: list[str]


@app.post("/predict/")
async def predict(inputs: Inputs = Body(...)):
    tok_out = tokenizer(inputs.items, padding='max_length', truncation=True, return_tensors='pt').to("cuda")
    logits = model(**tok_out).logits
    predictions = logits.argmax(dim=1).tolist()
    return {"predictions": predictions}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8000)
