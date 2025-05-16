import requests
from bs4 import BeautifulSoup
import json
import os
import re
import time
import sys
from urllib.parse import urljoin
import io
from PyPDF2 import PdfReader

def parse_si_page(si_url):
    """
    Fetches and parses an Irish Statutory Instrument page from irishstatutebook.ie.

    Args:
        si_url (str): The URL of the S.I. page (expected to be the direct HTML page).

    Returns:
        dict: A dictionary containing the extracted S.I. details (si_number, title, 
              pdf_url, publication_date, text_content), or None if parsing fails.
    """
    try:
        response = requests.get(si_url, timeout=10)
        response.raise_for_status()  # Raise an exception for bad status codes

        soup = BeautifulSoup(response.content, 'html.parser')

        si_details = {'url': si_url, 'pdf_url': None, 'text_content': '', 'text_source': ''}

        # Extract Title and S.I. Number (often in the <h1> tag)
        h1_tag = soup.find('h1')
        if h1_tag:
            full_title_text = h1_tag.get_text(separator=" ", strip=True)
            si_details['title'] = full_title_text
            match = re.search(r"S\.I\. No\. (\d+/\d+)", full_title_text)
            if match:
                si_details['si_number_full'] = match.group(0) # e.g., S.I. No. 123/2023
                si_details['si_number_simple'] = match.group(1) # e.g., 123/2023
            else:
                # Fallback: try to get from URL if possible, e.g., /eli/2023/si/123/made/en/print
                url_match = re.search(r"/si/(\d+)/made", si_url)
                if url_match:
                    year_match = re.search(r"/eli/(\d{4})/si/", si_url)
                    year = year_match.group(1) if year_match else "YYYY"
                    si_details['si_number_simple'] = f"{url_match.group(1)}/{year}"
                    si_details['si_number_full'] = f"S.I. No. {si_details['si_number_simple']}"
                else:
                    si_details['si_number_full'] = "Not found"
                    si_details['si_number_simple'] = "Not found"

        # Try to find the PDF link
        # For ELI /print pages, the title is usually 'Print this Document to PDF'
        pdf_link_tag = soup.find('a', title='Print this Document to PDF') 
        if pdf_link_tag and pdf_link_tag.has_attr('href'):
            si_details['pdf_url'] = urljoin(si_url, pdf_link_tag['href'])
        else:
            pdf_link_tag = soup.find('a', title='Download a PDF version of this SI')
            if pdf_link_tag and pdf_link_tag.has_attr('href'):
                si_details['pdf_url'] = urljoin(si_url, pdf_link_tag['href'])
            else:
                # Fallback: search all <a> tags for an href ending in .pdf
                for a_tag in soup.find_all('a', href=True): # Ensure href exists
                    href = a_tag['href']
                    # Check if href ends with .pdf (case-insensitive) and contains a slash (is a path)
                    if href.lower().endswith('.pdf') and '/' in href:
                        si_details['pdf_url'] = urljoin(si_url, href)
                        break # Take the first one found

        # Attempt PDF extraction first
        if si_details['pdf_url']:
            try:
                pdf_response = requests.get(si_details['pdf_url'], timeout=10)
                pdf_response.raise_for_status()
                
                with io.BytesIO(pdf_response.content) as pdf_file:
                    reader = PdfReader(pdf_file)
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text() or ""
                    
                    if text.strip():
                        si_details['text_content'] = text
                        si_details['text_source'] = 'pdf'
                    else:
                        print(f"  PDF text extraction resulted in empty content for {si_details['pdf_url']}. Will fall back to HTML.")

            except Exception as e_pdf:
                print(f"  PDF extraction failed for {si_details['pdf_url']}: {e_pdf}. Will fall back to HTML.")
                # Ensure pdf_url is nulled if extraction truly fails and we want to signify HTML was the final source
                # si_details['pdf_url'] = None # Or keep it to show attempt was made

        # Fallback to HTML parsing if PDF extraction failed or wasn't attempted
        if not si_details['text_content'] or si_details['text_source'] != 'pdf':
            html_text, content_node_debug_info = extract_text_from_html_content(soup, si_url)
            si_details['text_content'] = html_text
            si_details['text_source'] = 'html'
        
        # Ensure 'text_content' is not None for JSON serialization
        si_details['text_content'] = si_details['text_content'] or ""

        # Extract Publication Date
        # Priority 1: Try to find publication date from ELI metadata
        publication_date_from_meta = None
        # Example meta tag: <meta about="..." property="eli:date_document" CONTENT="YYYY-MM-DD" ... />
        meta_tag_for_date = soup.find('meta', attrs={'property': 'eli:date_document'})
        if meta_tag_for_date and meta_tag_for_date.get('content'):
            publication_date_from_meta = meta_tag_for_date['content']
        
        if publication_date_from_meta:
            si_details['publication_date'] = publication_date_from_meta
        else:
            # Fallback to searching the "Iris Oifigiúil" paragraph
            publication_notice_element = soup.find(lambda tag: tag.name == 'p' and "Iris Oifigiúil" in tag.get_text())
            if publication_notice_element:
                notice_text = publication_notice_element.get_text(separator=" ", strip=True)
                # Try to extract a date from this text, e.g., using a regex
                # This is a placeholder for a more robust date extraction if needed
                date_match = re.search(r'(\d{1,2}(?:st|nd|rd|th)? \w+ \d{4})', notice_text) # Basic date pattern
                if date_match:
                    si_details['publication_date'] = date_match.group(1)
                else:
                    si_details['publication_date'] = "Publication date not found in notice text (fallback)"
            else:
                si_details['publication_date'] = "Publication date not found (all methods failed)"

        return si_details
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL {si_url}: {e}")
        return None
    except Exception as e:
        print(f"An error occurred while parsing {si_url}: {e}")
        return None


def get_si_urls_for_year(year):
    """
    Fetches all S.I. 'print' URLs for a given year from irishstatutebook.ie.
    Args:
        year (int or str): The year to fetch S.I.s for.
    Returns:
        list: A list of S.I. URLs (print versions).
    """
    index_url = f"https://www.irishstatutebook.ie/eli/{year}/si/"
    si_urls = []
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    print(f"Fetching S.I. index for year {year} from {index_url}...")
    try:
        response = requests.get(index_url, headers=headers, timeout=10)
        response.raise_for_status()
        response.encoding = response.apparent_encoding # Let requests detect encoding
        print(f"Response received. Status: {response.status_code}. Content length: {len(response.text)} chars.")
        soup = BeautifulSoup(response.text, 'html.parser') 
        
        # New approach: Look for <tr> tags, then <a> tags with specific href pattern
        table_rows = soup.find_all('tr')
        if table_rows:
            for row in table_rows:
                # Find an <a> tag whose href matches the S.I. link pattern for the given year
                # e.g., /2023/en/si/0001.html or /2023/en/si/1.html
                link_pattern = re.compile(rf"^/{year}/en/si/\d+\.html$")
                a_tag = row.find('a', href=link_pattern)
                if a_tag:
                    relative_url = a_tag['href'] # e.g., /2023/en/si/123.html
                    # Extract S.I. number from the relative_url
                    si_match = re.search(r'/si/(\d+)\.html$', relative_url)
                    if si_match:
                        si_number_str = si_match.group(1) # Get the number part, e.g., '123' or '001'
                        # Construct the ELI print URL
                        # e.g., https://www.irishstatutebook.ie/eli/2023/si/1/made/en/print
                        # Note: S.I. number in ELI URL usually doesn't have leading zeros.
                        eli_page_url = f"https://www.irishstatutebook.ie/eli/{year}/si/{int(si_number_str)}/made/en/print"
                        if eli_page_url not in si_urls: # Avoid duplicates if any
                            si_urls.append(eli_page_url)
        else:
            print(f"Could not find any table rows (<tr>) on {index_url}")

    except requests.exceptions.RequestException as e:
        print(f"Error fetching S.I. index URL {index_url}: {e}")
    except Exception as e:
        print(f"An error occurred while parsing S.I. index {index_url}. Error details: {repr(e)}")
    
    print(f"Found {len(si_urls)} S.I. URLs for {year}.")
    return si_urls


def extract_text_from_html_content(soup, si_url):
    """
    Extracts the main text content from the given HTML soup.
    Args:
        soup (BeautifulSoup): The HTML soup to extract text from.
        si_url (str): The URL of the S.I. page (for debugging purposes).
    Returns:
        tuple: A tuple containing the extracted text and a debug info string about the content node used.
    """
    # Extract Title and S.I. Number (often in the <h1> tag)
    h1_tag = soup.find('h1')
    if h1_tag:
        full_title_text = h1_tag.get_text(separator=" ", strip=True)
        # ... (rest of the function remains the same)

if __name__ == "__main__":
    print("Crawler script starting...")
    
    TARGET_YEAR = 2023 # Example: Crawl S.I.s for 2023
    DATA_DIR = os.path.join("data", "crawled_docs")
    os.makedirs(DATA_DIR, exist_ok=True)

    print(f"Targeting S.I.s for the year: {TARGET_YEAR}")
    si_urls_for_year = get_si_urls_for_year(TARGET_YEAR)

    # Limit the number of S.I.s to process for this test run
    # TODO: Remove this limit for full crawl
    crawl_limit = None # or set to None to crawl all
    processed_count = 0

    for si_url in si_urls_for_year: # si_url is now an ELI print URL
        if crawl_limit is not None and processed_count >= crawl_limit:
            print(f"Reached crawl limit of {crawl_limit}. Stopping.")
            break

        print(f"\nFetching and parsing: {si_url}")
        si_data = parse_si_page(si_url) 
        if si_data:
            # Save the data
            # Ensure 'si_number_simple' is present for filename
            si_number_simple = si_data.get('si_number_simple', '').replace('/', '_')
            if not si_number_simple:
                print(f"  Skipping save for {si_url} due to missing S.I. number.")
                # Fallback to using a part of the ELI URL for filename if parsing failed to get S.I. number
                eli_url_match = re.search(r'/eli/(\d{4})/si/(\d+)/made', si_url)
                if eli_url_match:
                    year_from_eli = eli_url_match.group(1)
                    num_from_eli = eli_url_match.group(2)
                    si_number_simple = f"{num_from_eli}_{year_from_eli}"
                else: # Final fallback, less ideal
                    si_number_simple = f"unknown_eli_si_file_{processed_count}"
                print(f"  Using fallback filename: {si_number_simple}.json")

            filename = os.path.join(DATA_DIR, f"{si_number_simple}.json")
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(si_data, f, ensure_ascii=False, indent=4)
                print(f"  Successfully parsed and saved: {filename}")
            except IOError as e:
                print(f"  Error saving data for {si_url} to {filename}: {e}")
        else:
            print(f"  Failed to parse S.I. details for URL: {si_url}")
        processed_count += 1

    print(f"\nCrawler script finished. Processed {processed_count} S.I.(s).")
