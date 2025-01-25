import argparse
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.chrome.options import Options
import os
import requests
import subprocess
import time

keyword = ""
page_size = 100

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

        keyword = args.keyword
        wayback_url = "http://localhost:8080/cdc-archive-KEYWORD/record/".replace('KEYWORD', keyword)
        
        print(f"Starting downloading archive for: {keyword}")

        print(f"Get index: {query_url+keyword}")
        driver.get(query_url+keyword+f"&rows={page_size}")
        time.sleep(2)  # Wait for the initial page to fully load; adjust as necessary
        
        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        
        # Find all links with the class 'group'
        all_links = []
        pages = int(len(soup.find_all('li', class_='pagination-item')) / 2 - 2)  # the pagination exists two times and has last/next buttons
        links = [a.get('href') for a in soup.find_all('a', class_='group') if a.get('href')]
        all_links.extend(links)

        if pages > 1:
            for p in range(pages):
                print(query_url+keyword+f"&rows={page_size}&start="+str((p+1)*page_size))
                driver.get(query_url+keyword+f"&rows={page_size}&start="+str((p+1)*page_size))
                driver.refresh()
                time.sleep(2)
                
                soup = BeautifulSoup(driver.page_source, 'html.parser')
                links = [a.get('href') for a in soup.find_all('a', class_='group') if a.get('href')]
                all_links.extend(links)

        print("Archive index.")
        driver.get(wayback_url+query_url+keyword)
        driver.refresh()
        time.sleep(5)

        # Archive the content of each link
        for i, link in enumerate(all_links, start=1):
            if link.split('.')[-1] in ['pdf', 'xlsx']:
                file_url = base_url+'www_cdc_gov'+link.split('www.cdc.gov')[-1]
                file_name = link.split('/')[-1]
                response = requests.get(file_url, stream=True)
                    
                if response.status_code == 200:
                    with open(keyword+'/files/'+file_name, 'wb') as pdf_file:
                        pdf_file.write(response.content)
                    print(f"File successfully downloaded: {file_name}")
                else:
                    print(f"Failed to download file. Status code: {response.status_code}")
            else:
                archive_url = wayback_url+base_url+link
                print(f"Archive link: {archive_url}")

                driver.get(archive_url)
                driver.refresh()
                time.sleep(5)
    
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        driver.quit()
        print("Finished. Check your freshly created archive :)")

if __name__ == "__main__":
    main()