"""
Corpus Collection Tools for Japanese Text

This module provides data collection capabilities:
- Wikipedia Japanese dump downloader
- Web scraping utilities (news, blogs)
- Dataset integration (CC100, OSCAR)
- Text cleaning and normalization
"""

import os
import re
import json
import time
import requests
from typing import List, Dict, Optional, Generator
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import logging


class JapaneseCorpusCollector:
    """
    Corpus collector for Japanese text data.
    
    Collects text from various sources including Wikipedia,
    news websites, and open datasets.
    """
    
    def __init__(self, output_dir: str = "data/corpus"):
        self.output_dir = output_dir
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def collect_wikipedia_articles(self, max_articles: int = 1000) -> List[Dict]:
        """
        Collect Japanese Wikipedia articles.
        
        Args:
            max_articles: Maximum number of articles to collect
            
        Returns:
            List of collected articles with metadata
        """
        self.logger.info(f"Starting Wikipedia collection (max: {max_articles})")
        
        articles = []
        base_url = "https://ja.wikipedia.org"
        
        # Start with featured articles
        featured_url = f"{base_url}/wiki/特別:おすすめ記事"
        
        try:
            response = self.session.get(featured_url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find article links
            article_links = []
            for link in soup.find_all('a', href=True):
                href = link['href']
                if href.startswith('/wiki/') and ':' not in href:
                    article_links.append(urljoin(base_url, href))
            
            # Collect articles
            for i, article_url in enumerate(article_links[:max_articles]):
                if i % 10 == 0:
                    self.logger.info(f"Collected {i} articles...")
                
                article_data = self._collect_single_article(article_url)
                if article_data:
                    articles.append(article_data)
                
                # Rate limiting
                time.sleep(1)
                
        except Exception as e:
            self.logger.error(f"Error collecting Wikipedia articles: {e}")
        
        # Save collected articles
        self._save_collected_data(articles, "wikipedia_articles.json")
        
        self.logger.info(f"Collected {len(articles)} Wikipedia articles")
        return articles
    
    def _collect_single_article(self, url: str) -> Optional[Dict]:
        """Collect a single Wikipedia article."""
        try:
            response = self.session.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title = soup.find('h1', {'class': 'firstHeading'})
            title_text = title.get_text().strip() if title else ""
            
            # Extract main content
            content_div = soup.find('div', {'id': 'mw-content-text'})
            if not content_div:
                return None
            
            # Remove navigation and other non-content elements
            for element in content_div.find_all(['div', 'span'], class_=['navbox', 'infobox']):
                element.decompose()
            
            # Extract text content
            paragraphs = content_div.find_all('p')
            content_text = ""
            for p in paragraphs:
                text = p.get_text().strip()
                if text and len(text) > 50:  # Filter out short paragraphs
                    content_text += text + "\n"
            
            if len(content_text) < 200:  # Skip short articles
                return None
            
            return {
                'url': url,
                'title': title_text,
                'content': content_text,
                'word_count': len(content_text.split()),
                'char_count': len(content_text),
                'collected_at': time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Error collecting article {url}: {e}")
            return None
    
    def collect_news_articles(self, sources: List[str] = None) -> List[Dict]:
        """
        Collect Japanese news articles.
        
        Args:
            sources: List of news source URLs
            
        Returns:
            List of collected news articles
        """
        if sources is None:
            sources = [
                "https://www.nhk.or.jp/news/",
                "https://www.asahi.com/",
                "https://www.yomiuri.co.jp/",
                "https://www.mainichi.jp/"
            ]
        
        self.logger.info(f"Starting news collection from {len(sources)} sources")
        
        articles = []
        
        for source_url in sources:
            try:
                source_articles = self._collect_from_news_source(source_url)
                articles.extend(source_articles)
                self.logger.info(f"Collected {len(source_articles)} articles from {source_url}")
            except Exception as e:
                self.logger.error(f"Error collecting from {source_url}: {e}")
        
        # Save collected articles
        self._save_collected_data(articles, "news_articles.json")
        
        self.logger.info(f"Collected {len(articles)} news articles")
        return articles
    
    def _collect_from_news_source(self, source_url: str) -> List[Dict]:
        """Collect articles from a specific news source."""
        articles = []
        
        try:
            response = self.session.get(source_url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find article links (this is source-specific)
            article_links = self._extract_article_links(soup, source_url)
            
            for link in article_links[:50]:  # Limit per source
                article_data = self._collect_news_article(link)
                if article_data:
                    articles.append(article_data)
                time.sleep(0.5)  # Rate limiting
                
        except Exception as e:
            self.logger.error(f"Error processing {source_url}: {e}")
        
        return articles
    
    def _extract_article_links(self, soup: BeautifulSoup, source_url: str) -> List[str]:
        """Extract article links from news source page."""
        links = []
        
        # Common patterns for news article links
        link_selectors = [
            'a[href*="/news/"]',
            'a[href*="/article/"]',
            'a[href*="/story/"]',
            'a[href*="/topics/"]'
        ]
        
        for selector in link_selectors:
            for link in soup.select(selector):
                href = link.get('href')
                if href:
                    full_url = urljoin(source_url, href)
                    if self._is_valid_article_url(full_url):
                        links.append(full_url)
        
        return list(set(links))  # Remove duplicates
    
    def _is_valid_article_url(self, url: str) -> bool:
        """Check if URL is a valid article URL."""
        # Filter out non-article URLs
        invalid_patterns = [
            r'/category/',
            r'/tag/',
            r'/author/',
            r'/search',
            r'/login',
            r'/register',
            r'\.pdf$',
            r'\.jpg$',
            r'\.png$'
        ]
        
        for pattern in invalid_patterns:
            if re.search(pattern, url):
                return False
        
        return True
    
    def _collect_news_article(self, url: str) -> Optional[Dict]:
        """Collect a single news article."""
        try:
            response = self.session.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title = soup.find('h1') or soup.find('title')
            title_text = title.get_text().strip() if title else ""
            
            # Extract content (this is source-specific)
            content_text = self._extract_news_content(soup)
            
            if len(content_text) < 100:  # Skip short articles
                return None
            
            return {
                'url': url,
                'title': title_text,
                'content': content_text,
                'word_count': len(content_text.split()),
                'char_count': len(content_text),
                'collected_at': time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Error collecting news article {url}: {e}")
            return None
    
    def _extract_news_content(self, soup: BeautifulSoup) -> str:
        """Extract main content from news article."""
        # Common content selectors
        content_selectors = [
            'div.article-body',
            'div.content',
            'div.story-body',
            'div.article-content',
            'main article',
            'div.entry-content'
        ]
        
        for selector in content_selectors:
            content_div = soup.select_one(selector)
            if content_div:
                # Remove unwanted elements
                for element in content_div.find_all(['script', 'style', 'nav', 'aside']):
                    element.decompose()
                
                # Extract text
                paragraphs = content_div.find_all('p')
                content_text = ""
                for p in paragraphs:
                    text = p.get_text().strip()
                    if text:
                        content_text += text + "\n"
                
                if content_text:
                    return content_text
        
        # Fallback: extract all paragraphs
        paragraphs = soup.find_all('p')
        content_text = ""
        for p in paragraphs:
            text = p.get_text().strip()
            if text and len(text) > 20:
                content_text += text + "\n"
        
        return content_text
    
    def collect_blog_posts(self, max_posts: int = 500) -> List[Dict]:
        """
        Collect Japanese blog posts.
        
        Args:
            max_posts: Maximum number of blog posts to collect
            
        Returns:
            List of collected blog posts
        """
        self.logger.info(f"Starting blog collection (max: {max_posts})")
        
        # Japanese blog platforms
        blog_sources = [
            "https://ameblo.jp/",
            "https://blog.goo.ne.jp/",
            "https://blog.livedoor.jp/"
        ]
        
        posts = []
        
        for source in blog_sources:
            try:
                source_posts = self._collect_from_blog_source(source, max_posts // len(blog_sources))
                posts.extend(source_posts)
            except Exception as e:
                self.logger.error(f"Error collecting from blog source {source}: {e}")
        
        # Save collected posts
        self._save_collected_data(posts, "blog_posts.json")
        
        self.logger.info(f"Collected {len(posts)} blog posts")
        return posts
    
    def _collect_from_blog_source(self, source_url: str, max_posts: int) -> List[Dict]:
        """Collect posts from a specific blog source."""
        posts = []
        
        try:
            response = self.session.get(source_url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find blog post links
            post_links = self._extract_blog_links(soup, source_url)
            
            for link in post_links[:max_posts]:
                post_data = self._collect_blog_post(link)
                if post_data:
                    posts.append(post_data)
                time.sleep(0.5)  # Rate limiting
                
        except Exception as e:
            self.logger.error(f"Error processing blog source {source_url}: {e}")
        
        return posts
    
    def _extract_blog_links(self, soup: BeautifulSoup, source_url: str) -> List[str]:
        """Extract blog post links."""
        links = []
        
        # Common blog post link patterns
        link_selectors = [
            'a[href*="/entry/"]',
            'a[href*="/post/"]',
            'a[href*="/article/"]',
            'a[href*="/blog/"]'
        ]
        
        for selector in link_selectors:
            for link in soup.select(selector):
                href = link.get('href')
                if href:
                    full_url = urljoin(source_url, href)
                    if self._is_valid_article_url(full_url):
                        links.append(full_url)
        
        return list(set(links))
    
    def _collect_blog_post(self, url: str) -> Optional[Dict]:
        """Collect a single blog post."""
        try:
            response = self.session.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title = soup.find('h1') or soup.find('title')
            title_text = title.get_text().strip() if title else ""
            
            # Extract content
            content_text = self._extract_blog_content(soup)
            
            if len(content_text) < 100:  # Skip short posts
                return None
            
            return {
                'url': url,
                'title': title_text,
                'content': content_text,
                'word_count': len(content_text.split()),
                'char_count': len(content_text),
                'collected_at': time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Error collecting blog post {url}: {e}")
            return None
    
    def _extract_blog_content(self, soup: BeautifulSoup) -> str:
        """Extract main content from blog post."""
        # Common blog content selectors
        content_selectors = [
            'div.entry-content',
            'div.post-content',
            'div.article-content',
            'div.content',
            'main article',
            'div.entry-body'
        ]
        
        for selector in content_selectors:
            content_div = soup.select_one(selector)
            if content_div:
                # Remove unwanted elements
                for element in content_div.find_all(['script', 'style', 'nav', 'aside', 'footer']):
                    element.decompose()
                
                # Extract text
                paragraphs = content_div.find_all('p')
                content_text = ""
                for p in paragraphs:
                    text = p.get_text().strip()
                    if text:
                        content_text += text + "\n"
                
                if content_text:
                    return content_text
        
        # Fallback: extract all paragraphs
        paragraphs = soup.find_all('p')
        content_text = ""
        for p in paragraphs:
            text = p.get_text().strip()
            if text and len(text) > 20:
                content_text += text + "\n"
        
        return content_text
    
    def _save_collected_data(self, data: List[Dict], filename: str):
        """Save collected data to file."""
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Saved {len(data)} items to {filepath}")
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about collected data."""
        stats = {
            'wikipedia_articles': 0,
            'news_articles': 0,
            'blog_posts': 0,
            'total_text_length': 0
        }
        
        # Count files in output directory
        for filename in os.listdir(self.output_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(self.output_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            count = len(data)
                            if 'wikipedia' in filename:
                                stats['wikipedia_articles'] = count
                            elif 'news' in filename:
                                stats['news_articles'] = count
                            elif 'blog' in filename:
                                stats['blog_posts'] = count
                            
                            # Calculate total text length
                            for item in data:
                                if 'char_count' in item:
                                    stats['total_text_length'] += item['char_count']
                except Exception as e:
                    self.logger.error(f"Error reading {filename}: {e}")
        
        return stats
