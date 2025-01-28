import asyncio
import csv
import glob
import json
import os
import re
import time
import xml.etree.ElementTree as ET

from datetime import datetime

import aiofiles
import aiohttp
import requests
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.prompt import Prompt
from tqdm import tqdm

from selenium import webdriver
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.chrome.options import Options

console = Console()


class NLMDownloader:
    def __init__(self):
        self.base_url = "https://wsearch.nlm.nih.gov/ws/query"
        self.resource_base_url = "https://collections.nlm.nih.gov"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:133.0) Gecko/20100101 Firefox/133.0",
        }
        self.rate_limit_delay = 60 / 85  # 85 requests per minute max

        # WebDriver Setup
        options = Options()
        options.add_argument(
            "--headless"
        )  # Comment this line to see the browser in action
        options.page_load_strategy = "normal"

        self.driver = webdriver.Chrome(options=options)

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

    async def download_documents(self, ressource, output_dir, progress, task_id):
        """Download PDF"""
        try:
            filename = self.format_filename(ressource)
            pdf_path = os.path.join(output_dir, filename)

            if not ressource["pdf_url"]:
                progress.update(task_id, completed=100, total=100)
                console.print(
                    f"[yellow]No downloadable documents for {filename}.[/yellow]"
                )
                return False

            self.driver.get(ressource["pdf_url"])
            time.sleep(1)

            # todo: this process could be repeated for the ocr txt and the metadata files

            pdf_file_url = self.driver.current_url

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    pdf_file_url, headers=self.headers, allow_redirects=True
                ) as response:
                    if response.status == 200 or response.status == 202:
                        with open(pdf_path + ".pdf", "wb") as f:
                            while True:
                                chunk = await response.content.read(1024)
                                if not chunk:
                                    break
                                f.write(chunk)

                            progress.update(task_id, completed=100, total=100)
                            await asyncio.sleep(self.rate_limit_delay)
                            return True
                    elif response.status != 404:
                        console.print(
                            f"[red]Failed to download documents for {filename}: Status {response.status}[/red]"
                        )
                        return False

        except Exception as e:
            console.print(f"[red]Error downloading PDF for {filename}: {str(e)}[/red]")
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

    def format_filename(self, ressource):
        """Format filename for NLM ressources"""

        def clean_string(s):
            if not s:
                return ""
            s = s.replace('<span class="qt0">', "")  # todo: sorry...
            s = s.replace("</span>", "")
            s = re.sub(r'[<>:"/\\|?*]', "_", s)
            s = re.sub(r"[\s_]+", "_", s)
            s = "".join(c for c in s if ord(c) < 128)
            return s.strip("_")

        # Get components
        title = clean_string(ressource["title"])[:50]
        authors = "_".join([clean_string(a)[:20] for a in ressource["authors"]])[:50]
        date = ressource["date"]
        doc_id = ressource["id"].split("/")[-1]

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
            "Download research documents from the NLM Digital Collections",
            border_style="green",
        )
    )

    await download_from_nlm()


async def download_from_nlm():
    keywords = Prompt.ask("\n[yellow]Enter search keywords (comma-separated)[/yellow]")
    keywords = [k.strip() for k in keywords.split(",") if k.strip()]

    output_dir = Prompt.ask(
        "[yellow]Enter output directory[/yellow]", default="downloaded_papers_nlm"
    )
    os.makedirs(output_dir, exist_ok=True)

    downloader = NLMDownloader()

    # Define file paths for URLs and progress tracking
    urls_file = os.path.join(output_dir, "urls_list.csv")
    progress_file = os.path.join(output_dir, "download_progress.txt")

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

    # Check if we're resuming a previous session
    if os.path.exists(urls_file):
        resume = Prompt.ask(
            "\n[yellow]URL list found. Do you want to resume previous session? (yes/no)[/yellow]",
            choices=["yes", "no"],
            default="yes",
        )
        if resume.lower() == "yes":
            # Load existing progress
            downloaded_ids = set()
            if os.path.exists(progress_file):
                with open(progress_file, "r") as f:
                    downloaded_ids = set(f.read().splitlines())

            # Read existing URLs
            with open(urls_file, "r") as f:
                reader = csv.DictReader(f)
                urls_to_process = [row for row in reader]

            # Filter out already downloaded URLs
            urls_to_process = [
                url for url in urls_to_process if url["id"] not in downloaded_ids
            ]

            console.print(
                f"[green]Resuming download: {len(urls_to_process)} files remaining[/green]"
            )
        else:
            # Start fresh
            urls_to_process = await gather_urls(downloader, keywords, max_papers)
    else:
        # First run
        urls_to_process = await gather_urls(downloader, keywords, max_papers)

        # Save URLs to CSV file
        if urls_to_process:
            with open(urls_file, "w", newline="") as f:
                fieldnames = [
                    "id",
                    "title",
                    "authors",
                    "date",
                    "resource_url",
                    "pdf_url",
                    "txt_url",
                    "metadata_url",
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(urls_to_process)

    # Setup progress bars for downloading
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total})"),
        TimeRemainingColumn(),
        console=console,
    )

    # Download files using URL list
    with progress:
        download_task = progress.add_task(
            "[cyan]Overall download progress", total=len(urls_to_process)
        )

        for i, resource in enumerate(urls_to_process):
            try:
                file_task = progress.add_task(
                    f"[blue]Downloading {resource['title'][:50]}...", total=100
                )

                success = await downloader.download_documents(
                    resource, output_dir, progress, file_task
                )

                if success:
                    # Record successful download
                    with open(progress_file, "a") as f:
                        f.write(f"{resource['id']}\n")

                progress.update(download_task, completed=i + 1)
                progress.remove_task(file_task)

            except Exception as e:
                console.print(
                    f"[red]Error downloading {resource['id']}: {str(e)}[/red]"
                )
                continue

            await asyncio.sleep(1)  # Rate limiting

    # Create final index
    console.print("\n[yellow]Creating document index...[/yellow]")
    total_indexed = await downloader.create_index(output_dir)
    console.print(f"[green]Created index with {total_indexed} documents[/green]")

    # Final status
    downloaded_count = (
        len(set(open(progress_file).readlines()))
        if os.path.exists(progress_file)
        else 0
    )
    console.print(
        f"\n[green]Successfully downloaded {downloaded_count} documents to {output_dir}[/green]"
    )


async def gather_urls(downloader, keywords, max_papers):
    """Gather all URLs before downloading"""
    urls_to_process = []
    gathered = 0

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console,
    )

    with progress:
        gathering_task = progress.add_task("[cyan]Gathering URLs...", total=max_papers)

        for keyword in keywords:
            if gathered >= max_papers:
                break

            offset = 0
            while gathered < max_papers:
                results = await downloader.search_papers(
                    keyword, retstart=offset, retmax=100
                )

                if not results["documents"]:
                    break

                for document in results["documents"]:
                    if gathered >= max_papers:
                        break

                    try:
                        url = requests.get(document["url"], allow_redirects=True).url

                        pdf_url = txt_url = metadata_url = ""

                        if url.split("-")[-1] == "pdf":
                            pdf_url = url.replace("/catalog", "/pdfdownload")
                            txt_url = url.replace("/catalog", "/txt")
                        elif url.split("-")[-1] == "bk":
                            pdf_url = url.replace("/catalog", "/pdf")
                            txt_url = url.replace("/catalog", "/txt")
                        elif url.split("-")[-1] == "doc":
                            pdf_url = url.replace("/catalog", "/pdf")
                            txt_url = url.replace("/catalog", "/ocr")

                        metadata_url = url.replace("/catalog", "/dc")

                        resource = {
                            "id": document.get("identifier", "").split("/")[-1],
                            "title": document.get("title", ""),
                            "authors": "; ".join(document.get("authors", [])),
                            "date": document.get("date", ""),
                            "resource_url": document["url"],
                            "pdf_url": pdf_url,
                            "txt_url": txt_url,
                            "metadata_url": metadata_url,
                        }

                        urls_to_process.append(resource)
                        gathered += 1
                        progress.update(gathering_task, completed=gathered)

                    except Exception as e:
                        console.print(
                            f"[yellow]Error processing URL {document['url']}: {str(e)}[/yellow]"
                        )
                        continue

                offset += len(results["documents"])
                await asyncio.sleep(0.5)  # no bully the server :D

    return urls_to_process


if __name__ == "__main__":
    asyncio.run(main())
