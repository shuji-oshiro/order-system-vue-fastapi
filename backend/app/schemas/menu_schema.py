from .baseSchema import BaseSchema

class CategoryBase(BaseSchema):
    name: str
    description: str

class MenuOut(BaseSchema):
    id: int    
    category_id: int
    name: str
    price: int
    description: str
    search_text: str
    category: CategoryBase


class MenuOut_SP(BaseSchema):
    category_id: int
    category_name: str
    menues: list[MenuOut]


class MenuIn(BaseSchema):
    category_id: int
    name: str
    price: int
    description: str
    search_text: str

class MenuUpdate(BaseSchema):
    menu_id: int
    category_id: int
    name: str
    price: int
    description: str
    search_text: str


