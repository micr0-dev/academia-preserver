import requests
import json
import os
from tqdm import tqdm
import time
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
)
import re
import asyncio
import aiohttp
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
import json
import signal
import sys
from datetime import datetime
import xml.etree.ElementTree as ET
import glob

console = Console()


class DownloadState:
    def __init__(self, save_file="download_state.json"):
        self.save_file = save_file
        self.downloaded_papers = set()
        self.current_keyword = None
        self.current_offset = 0
        self.total_downloaded = 0
        self.keywords_completed = set()
        self.start_time = datetime.now().isoformat()
        self.keywords = []
        self.output_dir = "downloaded_papers"
        self.max_papers = None
        self.total_available = 0
        self.keyword_counts = {}

    def save(self):
        state = {
            "downloaded_papers": list(self.downloaded_papers),
            "current_keyword": self.current_keyword,
            "current_offset": self.current_offset,
            "total_downloaded": self.total_downloaded,
            "keywords_completed": list(self.keywords_completed),
            "start_time": self.start_time,
            "keywords": self.keywords,
            "output_dir": self.output_dir,
            "max_papers": self.max_papers,
            "total_available": self.total_available,
            "keyword_counts": self.keyword_counts,
        }
        with open(self.save_file, "w") as f:
            json.dump(state, f)

    def load(self):
        try:
            with open(self.save_file, "r") as f:
                state = json.load(f)
                self.downloaded_papers = set(state["downloaded_papers"])
                self.current_keyword = state["current_keyword"]
                self.current_offset = state["current_offset"]
                self.total_downloaded = state["total_downloaded"]
                self.keywords_completed = set(state["keywords_completed"])
                self.start_time = state["start_time"]
                self.keywords = state["keywords"]
                self.output_dir = state["output_dir"]
                self.max_papers = state["max_papers"]
                self.total_available = state["total_available"]
                self.keyword_counts = state["keyword_counts"]
                return True
        except FileNotFoundError:
            return False


class AcademiaDownloader:
    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:133.0) Gecko/20100101 Firefox/133.0",
            "Accept": "*/*",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br, zstd",
            "DNT": "1",
            "Connection": "keep-alive",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
        }
        self.session = requests.Session()

    def search_papers(self, keyword, offset=0, size=100):
        url = f"https://www.academia.edu/v0/search/integrated_search"
        params = {
            "camelize_keys": "true",
            "canonical": "true",
            "json": "true",
            "offset": offset,
            "query": f'"{keyword}"',
            "search_mode": "works",
            "size": size,
            "sort": "relevance",
            "subdomain_param": "api",
            "user_language": "en",
        }

        response = self.session.get(url, headers=self.headers, params=params)
        return response.json()

    def count_total_papers(self, keyword):
        """Count total available papers for the keyword"""
        results = self.search_papers(keyword, offset=0, size=1)
        return results.get("total", 0)

    def count_total_papers_multiple(self, keywords):
        """Count total available papers for multiple keywords"""
        total = 0
        for keyword in keywords:
            results = self.search_papers(keyword.strip(), offset=0, size=1)
            total += results.get("total", 0)
        return total

    def format_filename(self, work, attachment, unique_id, keyword):
        MAX_FILENAME_LENGTH = 200

        def clean_string(s):
            # Remove invalid characters and replace with underscore
            s = re.sub(r'[<>:"/\\|?*]', "_", s)
            # Replace multiple spaces/underscores with single underscore
            s = re.sub(r"[\s_]+", "_", s)
            # Remove non-ASCII characters
            s = "".join(c for c in s if ord(c) < 128)
            # Remove common words that don't add much value
            s = re.sub(r"_?(the|and|or|in|on|at|to|for|of|with)_?", "_", s.lower())
            return s.strip("_")

        paper_id = f"{work.get('id', '')}-{attachment.get('id', '')}"

        extension = os.path.splitext(attachment.get("fileName", ""))[1] or ".pdf"

        try:
            authors = work.get("authors", [])[:2]  # Only use first two authors
            author_names = []
            for author in authors:
                last_name = author.get("lastName", "").strip()
                if last_name:
                    author_names.append(clean_string(last_name))

            author_part = "_".join(author_names) if author_names else "Unknown"
            author_part = author_part[:30]  # Limit author part length

            # Clean and limit title
            title_part = clean_string(work.get("title", "Untitled"))
            title_part = title_part[:50]  # Limit title length

            keyword_part = clean_string(keyword)
            keyword_part = keyword_part[:20]  # Limit keyword length

            components = [author_part, title_part, keyword_part, unique_id, extension]

            filename = "-".join(c for c in components if c)

            if len(filename) > MAX_FILENAME_LENGTH:
                minimal_components = [
                    author_part[:20] if author_part else "unknown",
                    unique_id,
                    keyword_part[:10],
                    extension,
                ]
                filename = "-".join(c for c in minimal_components if c)

                if len(filename) > MAX_FILENAME_LENGTH:
                    filename = f"paper-{unique_id[:8]}{extension}"

            if len(filename) > MAX_FILENAME_LENGTH:
                raise ValueError(f"Filename still too long: {len(filename)} chars")

            return filename, paper_id

        except Exception as e:
            console.print(
                f"[yellow]Warning: Error creating filename: {str(e)}. Using fallback format.[/yellow]"
            )
            fallback = f"paper-{unique_id[:8]}{extension}"
            return fallback, paper_id

    async def download_paper_async(
        self, session, download_url, filename, output_dir, progress, file_task
    ):
        file_path = os.path.join(output_dir, filename)

        async with session.get(download_url, headers=self.headers) as response:
            total_size = int(response.headers.get("content-length", 0))
            downloaded_size = 0

            with open(file_path, "wb") as f:
                while True:
                    chunk = await response.content.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    progress.update(
                        file_task, completed=downloaded_size, total=total_size
                    )


class NLMDownloader:
    def __init__(self):
        self.base_url = "https://wsearch.nlm.nih.gov/ws/query"
        self.resource_base_url = "https://collections.nlm.nih.gov"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:133.0) Gecko/20100101 Firefox/133.0",
        }
        self.rate_limit_delay = 60 / 85  # 85 requests per minute max

    async def search_papers(self, keyword, retstart=0, retmax=100):
        """Search for papers with pagination"""
        params = {
            "db": "digitalCollections",
            "term": keyword,
            "retstart": retstart,
            "retmax": retmax,
        }

        async with aiohttp.ClientSession() as session:
            async with session.get(
                self.base_url, params=params, headers=self.headers
            ) as response:
                if response.status == 200:
                    xml_text = await response.text()
                    return self.parse_xml_response(xml_text)
                else:
                    raise Exception(f"Search failed with status {response.status}")

    def parse_xml_response(self, xml_text):
        """Parse XML response from NLM"""
        root = ET.fromstring(xml_text)
        results = {
            "count": int(root.find("count").text),
            "file": root.find("file").text if root.find("file") is not None else None,
            "server": (
                root.find("server").text if root.find("server") is not None else None
            ),
            "documents": [],
        }

        for doc in root.findall(".//document"):
            document = {
                "url": doc.get("url"),
                "rank": doc.get("rank"),
                "title": "",
                "authors": [],
                "date": "",
                "identifier": "",
            }

            for content in doc.findall("content"):
                name = content.get("name")
                if name == "dc:title":
                    document["title"] = content.text
                elif name == "dc:creator":
                    document["authors"].append(content.text)
                elif name == "dc:date":
                    document["date"] = content.text
                elif name == "dc:identifier":
                    document["identifier"] = content.text

            results["documents"].append(document)

        return results

    async def download_metadata(self, document, output_dir, nlm_id, suffix):
        """Download Dublin Core metadata"""
        try:
            # Create metadata URL
            metadata_url = f"{self.resource_base_url}/dc/nlm:nlmuid-{nlm_id}{suffix}"

            async with aiohttp.ClientSession() as session:
                async with session.get(metadata_url, headers=self.headers) as response:
                    if response.status == 200:
                        metadata_content = await response.text()

                        # Parse the XML metadata
                        root = ET.fromstring(metadata_content)

                        # Extract all metadata fields into a dictionary
                        metadata = {}
                        for child in root:
                            tag = child.tag.split("}")[-1]  # Remove namespace
                            if child.text:
                                if tag in metadata:
                                    if isinstance(metadata[tag], list):
                                        metadata[tag].append(child.text)
                                    else:
                                        metadata[tag] = [metadata[tag], child.text]
                                else:
                                    metadata[tag] = child.text

                        # Add source URL and download timestamp
                        metadata["source_url"] = document["url"]
                        metadata["download_timestamp"] = datetime.now().isoformat()
                        metadata["nlm_id"] = nlm_id
                        metadata["suffix_used"] = suffix

                        return metadata
            return None
        except Exception as e:
            console.print(f"[yellow]Error downloading metadata: {str(e)}[/yellow]")
            return None

    async def download_ocr_text(self, document, output_dir, progress, task_id):
        """Download OCR text version and metadata"""
        try:
            filename = self.format_filename(document)
            text_filename = os.path.splitext(filename)[0] + ".txt"
            metadata_filename = os.path.splitext(filename)[0] + "_metadata.json"
            text_path = os.path.join(output_dir, text_filename)
            metadata_path = os.path.join(output_dir, metadata_filename)

            # Extract ID and create OCR URL
            nlm_id = document["url"].split("/")[-1]

            # List of suffixes to try
            suffixes = ["-bk", "-doc"]

            async with aiohttp.ClientSession() as session:
                for suffix in suffixes:
                    ocr_url = (
                        f"{self.resource_base_url}/ocr/nlm:nlmuid-{nlm_id}{suffix}"
                    )

                    try:
                        async with session.get(
                            ocr_url, headers=self.headers
                        ) as response:
                            if response.status == 200:
                                # Download text content
                                try:
                                    content = await response.text(encoding="utf-8")
                                except UnicodeDecodeError:
                                    try:
                                        content = await response.text(
                                            encoding="latin-1"
                                        )
                                    except UnicodeDecodeError:
                                        raw_content = await response.read()
                                        content = raw_content.decode(
                                            "utf-8", errors="replace"
                                        )

                                # Download metadata
                                metadata = await self.download_metadata(
                                    document, output_dir, nlm_id, suffix
                                )

                                # Write text content
                                with open(
                                    text_path, "w", encoding="utf-8", errors="replace"
                                ) as f:
                                    f.write(content)

                                # Write metadata if available
                                if metadata:
                                    with open(
                                        metadata_path, "w", encoding="utf-8"
                                    ) as f:
                                        json.dump(
                                            metadata, f, indent=2, ensure_ascii=False
                                        )

                                progress.update(task_id, completed=100, total=100)
                                await asyncio.sleep(self.rate_limit_delay)
                                return True
                            elif response.status != 404:
                                console.print(
                                    f"[red]Failed to download OCR for {text_filename}: Status {response.status}[/red]"
                                )
                                return False
                    except Exception as e:
                        console.print(
                            f"[yellow]Error with {suffix} attempt for {text_filename}: {str(e)}[/yellow]"
                        )
                        continue

                console.print(
                    f"[red]Failed to download OCR for {text_filename} with all attempted suffixes[/red]"
                )
                return False

        except Exception as e:
            console.print(
                f"[red]Error downloading OCR for {text_filename}: {str(e)}[/red]"
            )
            return False

    async def create_index(self, output_dir):
        """Create an index of all downloaded documents"""
        index = []

        # Find all metadata files
        metadata_files = glob.glob(os.path.join(output_dir, "*_metadata.json"))

        for metadata_file in metadata_files:
            try:
                with open(metadata_file, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                    index.append(metadata)
            except Exception as e:
                console.print(
                    f"[yellow]Error reading metadata file {metadata_file}: {str(e)}[/yellow]"
                )

        # Save index
        index_path = os.path.join(output_dir, "_index.json")
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "total_documents": len(index),
                    "created_at": datetime.now().isoformat(),
                    "documents": index,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        return len(index)

    def format_filename(self, document):
        """Format filename for NLM documents"""

        def clean_string(s):
            if not s:
                return ""
            s = re.sub(r'[<>:"/\\|?*]', "_", s)
            s = re.sub(r"[\s_]+", "_", s)
            s = "".join(c for c in s if ord(c) < 128)
            return s.strip("_")

        # Get components
        title = clean_string(document.get("title", ""))[:50]
        authors = "_".join([clean_string(a)[:20] for a in document.get("authors", [])])[
            :50
        ]
        date = document.get("date", "")
        doc_id = document.get("identifier", "").split("/")[-1]

        # Create filename
        components = [p for p in [authors, title, date, doc_id] if p]
        filename = "-".join(components)

        # Ensure filename is not too long
        if len(filename) > 200:
            filename = f"{authors[:30]}-{doc_id}"

        return filename


async def main():
    console.print(
        Panel.fit(
            "[bold blue]Academic Paper Downloader[/bold blue]\n"
            "Download research papers from Academia.edu or NLM Digital Collections",
            border_style="green",
        )
    )

    # Choose source
    source = Prompt.ask(
        "\n[yellow]Choose source[/yellow]",
        choices=["academia", "nlm"],
        default="academia",
    )

    if source == "academia":
        # Use existing Academia.edu downloader
        await download_from_academia()
    else:
        # Use new NLM downloader
        await download_from_nlm()


async def download_from_nlm():
    keywords = Prompt.ask("\n[yellow]Enter search keywords (comma-separated)[/yellow]")
    keywords = [k.strip() for k in keywords.split(",") if k.strip()]

    output_dir = Prompt.ask(
        "[yellow]Enter output directory[/yellow]", default="downloaded_papers_nlm"
    )
    os.makedirs(output_dir, exist_ok=True)

    downloader = NLMDownloader()

    # Get total count for each keyword
    total_papers = 0
    keyword_counts = {}
    with console.status("[bold green]Counting available papers..."):
        for keyword in keywords:
            results = await downloader.search_papers(keyword, retstart=0, retmax=1)
            count = results["count"]
            keyword_counts[keyword] = count
            total_papers += count
            console.print(
                f"[green]Found {count} papers for keyword '{keyword}'[/green]"
            )

    console.print(f"[bold green]Total papers available: {total_papers}[/bold green]")

    max_papers = Prompt.ask(
        "[yellow]Maximum number of papers to download (or 'all' for no limit)[/yellow]",
        default="10",
    )
    max_papers = total_papers if max_papers.lower() == "all" else int(max_papers)

    # Setup progress bars
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total})"),
        TimeRemainingColumn(),
        console=console,
    )

    with progress:
        overall_task = progress.add_task("[cyan]Overall Progress", total=max_papers)

        downloaded = 0
        for keyword in keywords:
            if downloaded >= max_papers:
                break

            keyword_task = progress.add_task(
                f"[yellow]Progress for '{keyword}'",
                total=min(keyword_counts[keyword], max_papers - downloaded),
            )

            offset = 0
            while downloaded < max_papers:
                results = await downloader.search_papers(
                    keyword, retstart=offset, retmax=100
                )
                if not results["documents"]:
                    break

                for doc in results["documents"]:
                    if downloaded >= max_papers:
                        break

                    file_task = progress.add_task(
                        f"[blue]Downloading OCR for {doc['title'][:50]}...", total=100
                    )

                    success = await downloader.download_ocr_text(
                        doc, output_dir, progress, file_task
                    )

                    if success:
                        downloaded += 1
                        progress.update(overall_task, completed=downloaded)
                        progress.update(keyword_task, completed=downloaded)

                    progress.remove_task(file_task)

                offset += len(results["documents"])

            progress.remove_task(keyword_task)

    console.print("\n[yellow]Creating document index...[/yellow]")
    total_indexed = await downloader.create_index(output_dir)
    console.print(f"[green]Created index with {total_indexed} documents[/green]")

    console.print(
        f"\n[green]Successfully downloaded {downloaded} documents to {output_dir}[/green]"
    )


async def download_from_academia():
    console.print(
        Panel.fit(
            "[bold blue]Academia.edu Paper Downloader[/bold blue]\n"
            "Download research papers based on keywords",
            border_style="green",
        )
    )

    # Initialize download state
    state = DownloadState()

    downloader = AcademiaDownloader()

    new_session = True

    # Check for resume option first
    if state.load():
        resume = Prompt.ask(
            "\n[yellow]Previous download session found. Resume? (yes/no)[/yellow]",
            choices=["yes", "no"],
            default="yes",
        )
        if resume.lower() == "yes":
            console.print(
                f"[green]Resuming previous download session from {state.start_time}[/green]"
            )
            console.print(
                f"[green]Already downloaded: {state.total_downloaded} papers[/green]"
            )
            console.print(f"[green]Keywords: {', '.join(state.keywords)}[/green]")
            console.print(f"[green]Output directory: {state.output_dir}[/green]")
            console.print(
                f"[green]Total papers available: {state.total_available}[/green]"
            )
            for keyword, count in state.keyword_counts.items():
                console.print(f"[green]- {keyword}: {count} papers[/green]")
            return_to_main = Prompt.ask(
                "\n[yellow]Continue with these settings? (yes/no)[/yellow]",
                choices=["yes", "no"],
                default="yes",
            )
            if return_to_main.lower() == "yes":
                # Continue with loaded state

                if state.total_downloaded < state.max_papers:
                    state.keywords_completed.clear()

                if state.current_keyword:
                    keywords = [state.current_keyword] + [
                        k for k in state.keywords if k != state.current_keyword
                    ]
                else:
                    keywords = state.keywords

                output_dir = state.output_dir
                max_papers = state.max_papers
                total_available = state.total_available
                keyword_counts = state.keyword_counts
                new_session = False

                if state.total_downloaded >= state.max_papers:
                    console.print(
                        f"[green]Already downloaded maximum number of papers ({state.max_papers})\nPlease start a new session.[/green]"
                    )
                    return
            else:
                state = DownloadState()  # Reset state if not continuing
        else:
            state = DownloadState()  # Reset state if not resuming

    if new_session:
        keywords_input = Prompt.ask(
            "\n[yellow]Enter search keywords (comma-separated, or 'all' for everything)[/yellow]"
        )

        if keywords_input.lower() == "all":
            keywords = ["all"]
        else:
            keywords = [k.strip() for k in keywords_input.split(",") if k.strip()]

        state.keywords = keywords

        output_dir = Prompt.ask(
            "[yellow]Enter output directory[/yellow]", default="downloaded_papers"
        )
        state.output_dir = output_dir

        # Get total available papers for all keywords
        with console.status("[bold green]Counting total available papers..."):
            if keywords[0].lower() == "all":
                total_available = downloader.count_total_papers("all")
                state.total_available = total_available
                console.print(f"[green]Found {total_available} papers in total[/green]")
            else:
                keyword_counts = {}
                total_available = 0
                for keyword in keywords:
                    count = downloader.count_total_papers(keyword)
                    keyword_counts[keyword] = count
                    total_available += count
                    console.print(
                        f"[green]Found {count} papers for keyword '{keyword}'[/green]"
                    )
                state.keyword_counts = keyword_counts
                state.total_available = total_available
                console.print(
                    f"[bold green]Total papers across all keywords: {total_available}[/bold green]"
                )

        max_papers_input = Prompt.ask(
            "[yellow]Maximum number of papers to download (or 'all' for no limit)[/yellow]",
            default="10",
        )
        max_papers = (
            total_available
            if max_papers_input.lower() == "all"
            else int(max_papers_input)
        )
        state.max_papers = max_papers

    state.save()

    def signal_handler(sig, frame):
        console.print("\n[yellow]Saving progress and exiting...[/yellow]")
        state.save()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    os.makedirs(output_dir, exist_ok=True)

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total})"),
        TimeRemainingColumn(),
        console=console,
    )

    async with aiohttp.ClientSession() as session:
        with progress:
            overall_task = progress.add_task(
                "[cyan]Overall Progress",
                total=max_papers,
                completed=state.total_downloaded,
            )
            current_file_task = None

            try:
                for keyword in keywords:
                    if keyword in state.keywords_completed:
                        continue

                    state.current_keyword = keyword
                    offset = (
                        state.current_offset if keyword == state.current_keyword else 0
                    )

                    keyword_task = progress.add_task(
                        f"[yellow]Progress for '{keyword}'",
                        total=min(
                            keyword_counts.get(keyword, max_papers),
                            max_papers - state.total_downloaded,
                        ),
                    )

                    while True:
                        try:
                            results = downloader.search_papers(keyword, offset=offset)
                            works = results.get("works", [])

                            if not works:
                                break

                            for work in works:
                                if state.total_downloaded >= max_papers:
                                    break

                                attachments = work.get("downloadableAttachments", [])
                                if attachments:
                                    for attachment in attachments:
                                        if state.total_downloaded >= max_papers:
                                            break

                                        download_url = attachment.get("bulkDownloadUrl")
                                        if download_url:
                                            unique_id = str(uuid.uuid4())[:8]
                                            filename, paper_id = (
                                                downloader.format_filename(
                                                    work,
                                                    attachment,
                                                    unique_id,
                                                    keyword.strip(),
                                                )
                                            )

                                            if paper_id in state.downloaded_papers:
                                                continue

                                            if current_file_task is not None:
                                                progress.remove_task(current_file_task)

                                            current_file_task = progress.add_task(
                                                f"[blue]Downloading {filename[:30]}...",
                                                total=100,
                                            )

                                            await downloader.download_paper_async(
                                                session,
                                                download_url,
                                                filename,
                                                output_dir,
                                                progress,
                                                current_file_task,
                                            )

                                            state.downloaded_papers.add(paper_id)
                                            state.total_downloaded += 1
                                            progress.update(
                                                overall_task,
                                                completed=state.total_downloaded,
                                            )
                                            progress.update(
                                                keyword_task,
                                                completed=state.total_downloaded,
                                            )

                                            if state.total_downloaded % 10 == 0:
                                                state.save()

                                            time.sleep(0.2)  # Be nice to the server :)

                            offset += len(works)
                            state.current_offset = offset

                        except Exception as e:
                            console.print(
                                f"[red]Error with keyword '{keyword}': {str(e)}[/red]"
                            )
                            state.save()
                            break

                    if state.total_downloaded >= max_papers or not works:
                        state.keywords_completed.add(keyword)
                        state.save()
                    state.save()
                    progress.remove_task(keyword_task)

            except Exception as e:
                console.print(f"[red]Unexpected error: {str(e)}[/red]")
                state.save()

            finally:
                if current_file_task is not None:
                    progress.remove_task(current_file_task)

    console.print(
        f"\n[green]Successfully downloaded {state.total_downloaded} papers to {output_dir}[/green]"
    )

    if state.total_downloaded >= max_papers:
        try:
            os.remove(state.save_file)
        except:
            pass


if __name__ == "__main__":
    asyncio.run(main())
