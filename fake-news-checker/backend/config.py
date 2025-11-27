import os

from dotenv import load_dotenv

load_dotenv()


class Config:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", None)
    GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID", None)

    NEWS_API_KEY = os.getenv("NEWS_API_KEY", None)

    ENABLE_CACHE = os.getenv("ENABLE_CACHE", "true").lower() == "true"
    CACHE_TTL_HOURS = int(os.getenv("CACHE_TTL_HOURS", "24"))

    DEFAULT_NUM_RESULTS = int(os.getenv("DEFAULT_NUM_RESULTS", "5"))
    MAX_NUM_RESULTS = int(os.getenv("MAX_NUM_RESULTS", "10"))

    SIMILARITY_MODEL = os.getenv(
        "SIMILARITY_MODEL", "paraphrase-multilingual-MiniLM-L12-v2"
    )

    VERDICT_THRESHOLDS = {
        "HIGHLY_LIKELY_TRUE": 0.85,
        "LIKELY_TRUE": 0.70,
        "UNCERTAIN": 0.50,
        "LIKELY_FALSE": 0.30,
    }

    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", "8000"))
    API_RELOAD = os.getenv("API_RELOAD", "false").lower() == "true"

    # CORS settings
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    @classmethod
    def validate(cls):
        """Validate configuration và hiển thị warnings"""
        print("\n" + "=" * 70)
        print("🔧 CONFIGURATION STATUS")
        print("=" * 70)

        if cls.GOOGLE_API_KEY and cls.GOOGLE_CSE_ID:
            print(" Google Custom Search API: CONFIGURED")
            print("   → Will use Google API for searching (RECOMMENDED)")
        else:
            print("  Google Custom Search API: NOT CONFIGURED")
            print("   → Will use web scraping (may be less reliable)")
            print("   → Recommend: Setup Google API for better results")
            print("   → Visit: https://developers.google.com/custom-search/v1/overview")

        if cls.NEWS_API_KEY:
            print("NewsAPI: CONFIGURED")
        else:
            print("  NewsAPI: NOT CONFIGURED (optional)")

        print(f"\n Cache: {'ENABLED' if cls.ENABLE_CACHE else 'DISABLED'}")
        print(f" Cache TTL: {cls.CACHE_TTL_HOURS} hours")
        print(f" Default results: {cls.DEFAULT_NUM_RESULTS}")
        print(f" Similarity model: {cls.SIMILARITY_MODEL}")
        print(f" API Server: {cls.API_HOST}:{cls.API_PORT}")
        print("=" * 70 + "\n")

        return True


Config.validate()
