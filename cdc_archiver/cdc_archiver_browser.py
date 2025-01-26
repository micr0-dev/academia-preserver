import argparse
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.chrome.options import Options
import math
import os
import requests
import subprocess
import time

keyword = ""
page_size = 250

# Target URL
wayback_url = "http://localhost:8080/cdc-archive-KEYWORD/record/"
base_url    = "https://archive.cdc.gov/"
query_url   = base_url+"#/results?q="
details_url = base_url+"#/details?url="

# WebDriver Setup
options = Options()
options.add_argument("--headless")  # Comment this line to see the browser in action
options.page_load_strategy = 'normal'

driver = webdriver.Chrome(options=options)

def main():
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--keyword', type=str, help='Keyword')
        args = parser.parse_args()

        downloaded_documents = 0
        archived_websites = 0

        keyword = args.keyword
        wayback_url = "http://localhost:8080/cdc-archive-KEYWORD/record/".replace('KEYWORD', keyword)
        
        print(f"Starting downloading archive for: {keyword}")

        print("Creating index of all entries.")
        driver.get(query_url+keyword+f"&rows={page_size}")
        time.sleep(1)  # Wait for the initial page to fully load; adjust as necessary
        soup = BeautifulSoup(driver.page_source, 'html.parser')

        # Find all entries
        all_links = []
        entries   = int(soup.select_one('form + div strong').get_text())
        pages     = math.ceil(entries / page_size)
        links     = [a.get('href') for a in soup.find_all('a', class_='group') if a.get('href')]
        
        all_links.extend(links)

        if pages > 1:
            for p in range(pages):
                print(f"Still creating index. Results-Page {p+1} of {pages}.")
                driver.get(query_url+keyword+f"&rows={page_size}&start="+str((p+1)*page_size))
                driver.refresh()
                time.sleep(1)
                
                soup = BeautifulSoup(driver.page_source, 'html.parser')
                links = [a.get('href') for a in soup.find_all('a', class_='group') if a.get('href')]
                
                all_links.extend(links)

        print(f"Found {len(all_links)} entries.")

        # Archive the content of each link
        for i, link in enumerate(all_links, start=1):

            archive_url = base_url+'www_cdc_gov'+link.split('www.cdc.gov')[-1]

            if link.split('.')[-1] in ['pdf', 'xlsx', 'docx', 'pptx', 'xls', 'doc', 'ppt', 'csv']:
                file_name = archive_url.split('/')[-1]
                print(f"Downloading file {file_name}.")
                response = requests.get(archive_url, stream=True)
                    
                if response.status_code == 200:
                    with open(keyword+'/files/'+file_name, 'wb') as pdf_file:
                        pdf_file.write(response.content)
                    print(f"File successfully downloaded.")
                    downloaded_documents += 1
                else:
                    print(f"Failed to download file: {response.status_code}")

            else:
                print(f"Archiving link {archive_url}.")

                driver.get(archive_url)
                driver.refresh()
                time.sleep(2)
                archived_websites += 1
    
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        driver.quit()
        print("Finished. Check your freshly created archive.")
        print(f"Downloaded {downloaded_documents} documents, and archived {archived_websites} websites.")

if __name__ == "__main__":
    main()