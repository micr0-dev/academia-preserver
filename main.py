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
        """Format filename and return both filename and paper_id"""
        authors = work.get("authors", [])
        author_names = []
        for author in authors:
            full_name = (
                f"{author.get('firstName', '')} {author.get('lastName', '')}".strip()
            )
            if full_name:
                author_names.append(full_name)

        title = work.get("title", "Untitled")

        def clean_string(s):
            s = re.sub(r'[<>:"/\\|?*]', "_", s)
            s = re.sub(r"[\s_]+", "_", s)
            return s.strip("_")

        author_part = "_".join(author_names) if author_names else "Unknown_Author"
        author_part = clean_string(author_part)
        title_part = clean_string(title)
        extension = os.path.splitext(attachment.get("fileName", ""))[1]

        paper_id = f"{work.get('id', '')}-{attachment.get('id', '')}"

        filename = f"{author_part}-{title_part}-{keyword}-{unique_id}{extension}"

        if len(filename) > 225:
            max_title_length = (
                200
                - len(author_part)
                - len(keyword)
                - len(unique_id)
                - len(extension)
                - 4
            )
            if max_title_length > 0:
                title_part = title_part[:max_title_length]
            filename = f"{author_part}-{title_part}-{keyword}-{unique_id}{extension}"

        return filename, paper_id

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


async def main():
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
                keywords = [
                    k for k in state.keywords if k not in state.keywords_completed
                ]
                output_dir = state.output_dir
                max_papers = state.max_papers
                total_available = state.total_available
                keyword_counts = state.keyword_counts
                new_session = False
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

                    state.keywords_completed.add(keyword)
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
