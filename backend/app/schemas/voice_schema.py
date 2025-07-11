from pydantic import BaseModel
from .menu_schema import MenuOut

class MatchMenu(BaseModel):
    menu: MenuOut
    score: float

class VoiceResult(BaseModel):
    reco_text: str
    match_menus: list[MatchMenu]
