from fastapi import FastAPI
from pydantic import BaseModel
from question_answering import get_answer

app = FastAPI()

class Question(BaseModel):
    question: str

@app.post("/question")
async def answer_question(question: Question):
    answer = get_answer(question.question)
    return {"answer": answer}
   

    