import re
import unicodedata


def normalize_text(text: str) -> str:
    """
    Chuẩn hóa văn bản:
    - Chuyển về NFC
    - Chuyển về chữ thường
    - Loại bỏ các ký tự không cần thiết
    - Chuẩn hóa khoảng trắng
    """
    if not text:
        return ""

    text = unicodedata.normalize("NFC", text)
    text = text.lower()

    # Giữ lại các ký tự tiếng Việt, chữ số, và các dấu câu cơ bản
    text = re.sub(
        r"[^\w\s.,!?;:\-\(\)áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ]",
        " ",
        text,
    )

    text = re.sub(r"\s+", " ", text).strip()
    return text
