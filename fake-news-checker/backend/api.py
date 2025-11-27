import uvicorn
import os
import traceback
from contextlib import asynccontextmanager
from typing import Optional, Literal
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, model_validator, Field

from fact_checker import FactChecker

# --- Cấu hình Lifespan (Thay thế cho @app.on_event) ---
# Khởi tạo một biến toàn cục cho FactChecker
fact_checker_instance = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Quản lý vòng đời của ứng dụng FastAPI.
    Khởi tạo FactChecker khi server khởi động.
    """
    global fact_checker_instance
    print("Starting up Fact Checker API...")
    fact_checker_instance = FactChecker()
    print("Fact Checker initialized successfully!")
    yield
    # (Có thể thêm code dọn dẹp ở đây nếu cần)
    print("Shutting down API...")


# Khởi tạo FastAPI app với lifespan
app = FastAPI(
    title="Fake News Detection API",
    description="API để phát hiện tin giả trên mạng xã hội",
    version="1.0.1",  # Tăng version
    lifespan=lifespan,  # Sử dụng lifespan manager mới
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Trong thực tế nên giới hạn lại
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Pydantic Models ---


class FactCheckRequest(BaseModel):
    """Request model cho việc kiểm tra tin tức."""

    content: str
    input_type: Literal["text", "url"] = "text"
    num_sources: Optional[int] = Field(
        default=5, ge=1, le=10, description="Số lượng nguồn tham khảo (1-10)"
    )

    @model_validator(mode="after")
    def validate_content_based_on_type(self):
        """
        Validator tùy chỉnh để kiểm tra content dựa trên input_type.

        """
        content = self.content
        input_type = self.input_type

        if not content or not content.strip():
            raise ValueError("Content không được để trống")

        if input_type == "text":
            word_count = len(content.split())
            if word_count < 3:
                raise ValueError("Content quá ngắn (tối thiểu 3 từ)")

        self.content = content.strip()
        return self


class FactCheckResponse(BaseModel):
    """Response model chuẩn cho kết quả kiểm tra."""

    success: bool
    message: Optional[str] = None
    verdict: Optional[dict] = None
    references: Optional[list] = None
    keywords: Optional[list] = None
    timestamp: Optional[str] = None


# --- API Endpoints ---


@app.get("/", tags=["Health"])
async def root():
    """Health check endpoint cơ bản."""
    return {
        "status": "online",
        "message": "Fake News Detection API is running",
        "version": "1.0.1",
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check chi tiết."""
    return {
        "status": "healthy",
        "fact_checker_initialized": fact_checker_instance is not None,
        "endpoints": {"check": "/api/check", "health": "/health"},
    }


@app.post("/api/check", response_model=FactCheckResponse, tags=["Core"])
async def check_fact(request: FactCheckRequest):
    """
    Endpoint chính để kiểm tra độ tin cậy của tin tức.
    Nhận vào text hoặc URL và trả về phán quyết.
    """
    global fact_checker_instance
    try:
        if fact_checker_instance is None:
            raise HTTPException(
                status_code=503, detail="Fact checker chưa được khởi tạo"
            )

        print(f"\n{'='*60}")
        print(f"[API] New request:")
        print(f"  Type: {request.input_type}")
        print(f"  Content: {request.content[:100]}...")
        print(f"{'='*60}\n")

        # Chạy pipeline kiểm tra
        result = fact_checker_instance.check_fact(
            user_input=request.content,
            input_type=request.input_type,
            num_sources=request.num_sources,
        )

        print(f"\n[API] Result status: {result['status']}")

        # Format kết quả cho frontend
        formatted_result = fact_checker_instance.format_result_for_frontend(result)

        return JSONResponse(content=formatted_result)

    except ValueError as e:
        # Lỗi 400 cho các vấn đề validation (ví dụ: text quá ngắn)
        print(f"[API] ValueError: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Lỗi 500 cho các lỗi server nội bộ
        print(f"[API] Exception: {type(e).__name__} - {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý nội bộ: {str(e)}")


@app.get("/api/trusted-sources", tags=["Utility"])
async def get_trusted_sources():
    """Lấy danh sách các nguồn tin uy tín đang được sử dụng."""
    global fact_checker_instance
    if fact_checker_instance is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    if hasattr(fact_checker_instance, "searcher") and hasattr(
        fact_checker_instance.searcher, "trusted_sources"
    ):
        return {
            "sources": list(fact_checker_instance.searcher.trusted_sources.keys()),
            "count": len(fact_checker_instance.searcher.trusted_sources),
        }
    raise HTTPException(status_code=500, detail="Source list not available")


# Main entry point
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=False, log_level="info")
