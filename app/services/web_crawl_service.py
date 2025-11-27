"""Web crawling service for extracting and indexing web page content."""

import json
import logging
import asyncio
from typing import List, Optional
from urllib.parse import urlparse
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.models import CrawlResultContainer
import requests
from bs4 import BeautifulSoup
from sqlalchemy.orm import Session
from sqlalchemy import select
from llama_index.core import Document
from app.models import Embedding, WebCrawlSource, KnowledgeBase
import datetime

from app.services.ingest_service import IngestService

logger = logging.getLogger(__name__)


class WebCrawlService:
    """Service for crawling web pages and extracting content."""

    def __init__(self, db: Session):
        self.db = db
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
        )
        self.crawler = AsyncWebCrawler()
        self.ingest_service = IngestService(db=db)

    async def crawl_url(self, url: str) -> dict:
        """Crawl a single URL and extract content.

        Args:
            url: URL to crawl

        Returns:
            Dictionary with 'title', 'text', and 'error' keys
        """
        try:
            logger.info(f"Crawling URL: {url}")

            # Fetch the page (run in thread pool to avoid blocking)
            response = await asyncio.to_thread(
                self.session.get, url, timeout=30, allow_redirects=True
            )
            response.raise_for_status()

            # Parse HTML
            soup = await asyncio.to_thread(
                BeautifulSoup, response.content, "html.parser"
            )

            # Extract title
            title = None
            if soup.title:
                title = soup.title.get_text().strip()
            elif soup.find("meta", property="og:title"):
                title = (
                    soup.find("meta", property="og:title").get("content", "").strip()
                )
            elif soup.find("h1"):
                title = soup.find("h1").get_text().strip()

            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
                script.decompose()

            # Extract main content
            # Try to find main content area
            main_content = None
            for selector in [
                "main",
                "article",
                '[role="main"]',
                ".content",
                "#content",
            ]:
                main_content = soup.select_one(selector)
                if main_content:
                    break

            if main_content:
                text = main_content.get_text(separator="\n", strip=True)
            else:
                # Fallback to body
                body = soup.find("body")
                if body:
                    text = body.get_text(separator="\n", strip=True)
                else:
                    text = soup.get_text(separator="\n", strip=True)

            # Clean up text (remove excessive whitespace)
            lines = [line.strip() for line in text.split("\n") if line.strip()]
            text = "\n".join(lines)

            logger.info(
                f"Successfully crawled URL: {url} (Title: {title}, Content length: {len(text)})"
            )

            return {"title": title or url, "text": text, "error": None}

        except requests.exceptions.RequestException as e:
            error_msg = f"Failed to fetch URL {url}: {str(e)}"
            logger.error(error_msg)
            return {"title": None, "text": None, "error": error_msg}
        except Exception as e:
            error_msg = f"Error processing URL {url}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {"title": None, "text": None, "error": error_msg}

    async def _crawl_url(self, url: str):
        """Crawl a single URL and extract content.

        Args:
            url: URL to crawl

        Returns:
            Dictionary with 'title', 'text', and 'error' keys
        """
        try:

            crawl_config = CrawlerRunConfig(
                wait_for="body",
                delay_before_return_html=1,
            )

            crawl_result = await self.crawler.arun(url=url, config=crawl_config)

            serialized_result = [result.model_dump() for result in crawl_result]

            with open("crawl_result.json", "w") as f:
                json.dump(serialized_result, f, indent=4)
                logger.info(f"Crawl result saved to crawl_result.json")

            return {
                "title": crawl_result[0].metadata.get("title"),
                "markdown": crawl_result[0].markdown.markdown_with_citations,
                "metadata": crawl_result[0].metadata,
                "error": None,
            }
        except Exception as e:
            error_msg = f"Error crawling URL {url}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {"title": None, "html": None, "metadata": None, "error": error_msg}

    async def add_urls_to_kb(
        self, kb_id: str, urls: List[str], descriptions: Optional[List[str]] = None
    ) -> List[WebCrawlSource]:
        """Add multiple URLs to a knowledge base.

        Args:
            kb_id: Knowledge base ID
            urls: List of URLs to add
            descriptions: Optional list of descriptions for each URL

        Returns:
            List of created WebCrawlSource objects
        """
        from uuid import UUID

        kb_uuid = UUID(kb_id)

        # Verify KB exists
        kb_result = self.db.execute(
            select(KnowledgeBase).where(KnowledgeBase.id == kb_uuid)
        )
        kb = kb_result.scalar_one_or_none()
        if not kb:
            raise ValueError(f"Knowledge base {kb_id} not found")

        created_sources = []
        descriptions = descriptions or [None] * len(urls)

        for url, description in zip(urls, descriptions):
            # Normalize URL
            url = url.strip()
            if not url.startswith(("http://", "https://")):
                url = "https://" + url

            # Check if URL already exists for this KB
            existing = self.db.execute(
                select(WebCrawlSource).where(
                    WebCrawlSource.kb_id == kb_uuid, WebCrawlSource.url == url
                )
            ).scalar_one_or_none()

            if existing:
                logger.warning(f"URL {url} already exists for KB {kb_id}, skipping")
                continue

            # Create new source
            source = WebCrawlSource(
                kb_id=kb_uuid,
                url=url,
                description=description.strip() if description else None,
                is_indexed=False,
            )
            self.db.add(source)
            created_sources.append(source)

        self.db.commit()

        # Refresh objects
        for source in created_sources:
            self.db.refresh(source)

        logger.info(f"Added {len(created_sources)} URLs to KB {kb_id}")
        return created_sources

    async def crawl_and_index_url(
        self,
        source_id: str,
    ) -> dict:
        """Crawl a URL and index its content.

        Args:
            source_id: WebCrawlSource ID

        Returns:
            Dictionary with success status and message
        """
        from uuid import UUID

        source_uuid = UUID(source_id)

        # Get the source
        result = self.db.execute(
            select(WebCrawlSource).where(WebCrawlSource.id == source_uuid)
        )
        source = result.scalar_one_or_none()
                
                        
        self.db.query(Embedding).filter(
            Embedding.chunk_metadata.op("->>")("source_id") == str(source_id)
        ).delete()
                
        if not source:
            return {"success": False, "error": "Source not found"}

        try:
            # Crawl the URL
            crawl_result = await self._crawl_url(url=source.url)

            if crawl_result["error"]:
                return {"success": False, "error": crawl_result["error"]}

            # Update source with title
            if crawl_result["title"]:
                source.title = crawl_result["title"]
                self.db.flush()

            await self.ingest_service.process_markdown(
                kb_id=source.kb_id,
                markdown=crawl_result["markdown"],
                source=source,
            )

            # Update source status
            source.is_indexed = True
            source.last_crawled_at = datetime.datetime.now(datetime.UTC)
            self.db.commit()

            return {
                "success": True,
                "message": f"Successfully crawled and indexed {source.url}",
            }

        except Exception as e:
            logger.error(
                f"Error crawling and indexing URL {source.url}: {str(e)}", exc_info=True
            )
            self.db.rollback()
            return {"success": False, "error": str(e)}

    async def crawl_and_index_multiple_urls(
        self,
        source_ids: List[str],
    ) -> dict:
        """Crawl and index multiple URLs.

        Args:
            source_ids: List of WebCrawlSource IDs

        Returns:
            Dictionary with results for each URL
        """
        results = {}

        for source_id in source_ids:
            result = await self.crawl_and_index_url(source_id)
            results[source_id] = result

        return results
