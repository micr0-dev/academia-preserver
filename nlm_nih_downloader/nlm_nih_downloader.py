import asyncio
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
from rich.progress import (BarColumn, Progress, SpinnerColumn, TextColumn,
                           TimeRemainingColumn)
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
        options.add_argument("--headless")  # Comment this line to see the browser in action
        options.page_load_strategy = 'normal'

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

    async def download_documents(self, document, output_dir, progress, task_id):
        """Download PDF"""
        try:
            filename = self.format_filename(document)
            pdf_path = os.path.join(output_dir, filename)
            document["url"].split("/")[-1]
            url      = requests.get(document["url"], allow_redirects=True).url # grab redirect

            if url.split('-')[-1] not in ['pdf', 'bk', 'doc']:
                progress.update(task_id, completed=100, total=100)
                console.print(
                    f"[yellow]No downloadable documents for {filename}.[/yellow]"
                )
                return False

            if url.split('-')[-1] == 'pdf':
                pdf_url = url.replace("/catalog", "/pdfdownload")
                url.replace("/catalog", "/txt")
            
            if url.split('-')[-1] == 'bk':
                pdf_url = url.replace("/catalog", "/pdf")
                url.replace("/catalog", "/txt")
            
            if url.split('-')[-1] == 'doc':
                pdf_url = url.replace("/catalog", "/pdf")
                url.replace("/catalog", "/ocr")
            
            url.replace("/catalog", "/dc")

            self.driver.get(pdf_url)
            time.sleep(1)

            # todo: this process could be repeated for the ocr txt and the metadata files

            pdf_file_url = self.driver.current_url

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    pdf_file_url, headers=self.headers, allow_redirects=True
                ) as response:
                    if response.status == 200 or response.status == 202:
                        with open(pdf_path+'.pdf', 'wb') as f:
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
            console.print(
                f"[red]Error downloading PDF for {pdf_url}: {str(e)}[/red]"
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
            s = s.replace('<span class="qt0">', '') # todo: sorry...
            s = s.replace('</span>', '')
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
                        f"[blue]Downloading documents for {doc['title'][:50]}...", total=100
                    )

                    # success_ocr = await downloader.download_ocr_text(
                    #     doc, output_dir, progress, file_task
                    # )

                    success_pdf = await downloader.download_documents(
                        doc, output_dir, progress, file_task
                    )

                    if success_pdf:
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


if __name__ == "__main__":
    asyncio.run(main())
