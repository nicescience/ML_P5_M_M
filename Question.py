from pydantic import BaseModel
from pydantic import ValidationError


class Question(BaseModel):
    question :str

    