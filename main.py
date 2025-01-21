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

console = Console()


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

    def format_filename(self, work, attachment, unique_id):
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
            return re.sub(r'[<>:"/\\|?*]', "_", s)

        author_part = "_".join(author_names) if author_names else "Unknown_Author"
        title_part = clean_string(title)
        extension = os.path.splitext(attachment.get("fileName", ""))[1]

        filename = f"{author_part}-{title_part}-{unique_id}{extension}"

        if len(filename) > 255:
            filename = filename[:240] + f"-{unique_id}{extension}"

        return filename

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

    keywords_input = Prompt.ask(
        "\n[yellow]Enter search keywords (comma-separated, or 'all' for everything)[/yellow]"
    )

    if keywords_input.lower() == "all":
        keywords = ["all"]
    else:
        keywords = [k.strip() for k in keywords_input.split(",") if k.strip()]

    output_dir = Prompt.ask(
        "[yellow]Enter output directory[/yellow]", default="downloaded_papers"
    )

    downloader = AcademiaDownloader()

    with console.status("[bold green]Counting total available papers..."):
        if keywords[0].lower() == "all":
            total_available = downloader.count_total_papers("all")
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
            console.print(
                f"[bold green]Total papers across all keywords: {total_available}[/bold green]"
            )

    max_papers_input = Prompt.ask(
        "[yellow]Maximum number of papers to download (or 'all' for no limit)[/yellow]",
        default="10",
    )
    max_papers = (
        total_available if max_papers_input.lower() == "all" else int(max_papers_input)
    )

    concurrent_downloads = int(
        Prompt.ask("[yellow]Number of concurrent downloads[/yellow]", default="3")
    )

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
            overall_task = progress.add_task("[cyan]Overall Progress", total=max_papers)
            current_file_task = None
            downloaded = 0

            for keyword in keywords:
                if downloaded >= max_papers:
                    break

                offset = 0
                keyword_task = progress.add_task(
                    f"[yellow]Progress for '{keyword}'",
                    total=(
                        min(
                            keyword_counts.get(keyword, max_papers),
                            max_papers - downloaded,
                        )
                        if keywords[0].lower() != "all"
                        else max_papers
                    ),
                )
                keyword_downloaded = 0

                while True:
                    try:
                        results = downloader.search_papers(keyword, offset=offset)
                        works = results.get("works", [])

                        if not works:
                            break

                        for work in works:
                            if downloaded >= max_papers:
                                break

                            attachments = work.get("downloadableAttachments", [])
                            if attachments:
                                for attachment in attachments:
                                    if downloaded >= max_papers:
                                        break

                                    download_url = attachment.get("bulkDownloadUrl")
                                    if download_url:
                                        unique_id = str(uuid.uuid4())[:8]
                                        filename = downloader.format_filename(
                                            work, attachment, unique_id
                                        )

                                        base, ext = os.path.splitext(filename)
                                        filename = f"{base}__{keyword.strip()}{ext}"

                                        if current_file_task is not None:
                                            progress.remove_task(current_file_task)

                                        current_file_task = progress.add_task(
                                            f"[blue]Downloading {filename[:30]}",
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

                                        downloaded += 1
                                        keyword_downloaded += 1
                                        progress.update(
                                            overall_task, completed=downloaded
                                        )
                                        progress.update(
                                            keyword_task, completed=keyword_downloaded
                                        )
                                        time.sleep(0.5)  # Be nice to the server :)

                        offset += len(works)

                    except Exception as e:
                        console.print(
                            f"[red]Error with keyword '{keyword}': {str(e)}[/red]"
                        )
                        break

                progress.remove_task(keyword_task)

            if current_file_task is not None:
                progress.remove_task(current_file_task)

    console.print(
        f"\n[green]Successfully downloaded {downloaded} papers to {output_dir}[/green]"
    )


if __name__ == "__main__":
    asyncio.run(main())
