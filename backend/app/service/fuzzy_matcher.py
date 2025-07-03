import pykakasi
from rapidfuzz import fuzz
from backend.app.crud import menu_crud
from backend.app.database.database import get_db
from backend.app.schemas.menu_schema import MenuOut


def to_hira(text: str) -> str:
    kakasi = pykakasi.kakasi()
    text_hira = kakasi.convert(text)
    return "".join(item["hira"] for item in text_hira)

def fuzzy_menu_match(text: str, threshold: int = 50):
    cnv_text = to_hira(text)
    
    result = menu_crud.get_menus(next(get_db()))
    menu_out_list = [MenuOut.model_validate(item) for item in result]
    # 各メニューの検索用テキストの配列をキーにメニューオブジェクトのタプルを作成
    search_menus = [(menu.search_text, menu) for menu in menu_out_list]

    match_menus = []
    for search_text, menu in search_menus:
        score = fuzz.ratio(cnv_text, search_text)
        if score >= threshold:
            match_menus.append((menu, score))

    return match_menus
