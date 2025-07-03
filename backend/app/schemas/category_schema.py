from .baseSchema import BaseSchema

class CategoryOut(BaseSchema):
    id: int
    name: str
    description: str

class CategoryIn(BaseSchema):
    name: str
    description: str