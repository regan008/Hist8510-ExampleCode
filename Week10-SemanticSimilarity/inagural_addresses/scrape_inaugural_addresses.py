#!/usr/bin/env python3
"""
Scraper for Inaugural Addresses from American Presidency Project

This script scrapes the text of presidential inaugural addresses from the
UCSB American Presidency Project website.
Website: https://www.presidency.ucsb.edu/documents/app-categories/spoken-addresses-and-remarks/presidential/inaugural-addresses

Author: Created for History 8510 at Clemson University
"""

import requests
from bs4 import BeautifulSoup
import re
import time
import os
from urllib.parse import urljoin
import csv

BASE_URL = "https://www.presidency.ucsb.edu"
LISTING_URL = "https://www.presidency.ucsb.edu/documents/app-categories/spoken-addresses-and-remarks/presidential/inaugural-addresses"

def get_page_with_retry(url, max_retries=3):
    """Fetch a webpage with retry logic."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            if attempt < max_retries - 1:
                print(f"Attempt {attempt + 1} failed: {e}. Retrying...")
                time.sleep(2)
            else:
                print(f"Failed after {max_retries} attempts: {e}")
                return None
    
    return None

def extract_president_and_date(soup, url):
    """Extract president name and date from the page."""
    # Try to find the title which usually contains president name and date
    president = "Unknown"
    date = "Unknown"
    
    # Look for title
    title_tag = soup.find('h1')
    if title_tag:
        title_text = title_tag.get_text(strip=True)
        
        # Try to extract date from various formats
        date_pattern = r'(\w+day,?\s+\w+\s+\d{1,2},?\s+\d{4})|(\w+\s+\d{1,2},?\s+\d{4})'
        date_match = re.search(date_pattern, title_text)
        if date_match:
            date = date_match.group(0).strip()
    
    # Look for president name in the page
    # The president name is often in a specific section
    president_tags = soup.find_all(['h2', 'h3', 'h4'])
    for tag in president_tags:
        text = tag.get_text(strip=True)
        # Check if this looks like a president's name
        if re.search(r'^(President\s+)?[A-Z][a-z]+\s+[A-Z][a-z]+\s*(\([^)]+\))?$', text):
            president = text
            break
    
    # Try alternative: look for "by" followed by a name
    if president == "Unknown":
        for tag in soup.find_all(['p', 'div']):
            text = tag.get_text()
            match = re.search(r'by\s+([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)', text)
            if match:
                president = match.group(1)
                break
    
    return president, date

def extract_address_text(soup):
    """Extract the actual inaugural address text from the page."""
    # The address text is typically in a div with class containing "displaytext"
    # or in a field-docs-content div
    text_content = ""
    
    # Try multiple selectors that are commonly used
    selectors = [
        'div.field-docs-content',
        'div.displaytext',
        'div.docs-content',
        'div.field-ds-doc-text',
    ]
    
    for selector in selectors:
        content_div = soup.select_one(selector)
        if content_div:
            # Get all the text from this div, preserving paragraph structure
            paragraphs = content_div.find_all('p')
            if paragraphs:
                for p in paragraphs:
                    text = p.get_text(strip=True)
                    if text:
                        text_content += text + "\n\n"
                break
            else:
                # If no paragraphs, just get all text
                text_content = content_div.get_text(separator='\n', strip=True)
                break
    
    # If that didn't work, try to find the main content area
    if not text_content:
        # Look for article or main content
        main_content = soup.find('article') or soup.find('main')
        if main_content:
            # Get text but skip headers and metadata
            for element in main_content.find_all(['p', 'div']):
                if element.get('class') and any('meta' in str(c).lower() for c in element.get('class')):
                    continue
                text = element.get_text().strip()
                if text and len(text) > 50:  # Filter out very short texts
                    text_content += text + "\n\n"
    
    # Clean up the text
    text_content = re.sub(r'\n\s*\n\s*\n+', '\n\n', text_content)  # Remove excessive newlines
    text_content = text_content.strip()
    
    return text_content

def get_address_links(max_pages=2):
    """Get all links to inaugural addresses from the listing page(s)."""
    all_addresses = []
    
    # Get pagination URLs
    for page in range(max_pages):
        if page == 0:
            url = LISTING_URL
        else:
            # Check if there's a page= parameter needed
            url = f"{LISTING_URL}?page={page}"
        
        print(f"Fetching page {page + 1}/{max_pages}...")
        response = get_page_with_retry(url)
        
        if not response:
            print(f"Failed to fetch page {page + 1}")
            continue
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find all links to individual addresses
        addresses = []
        
        # The inaugural addresses are in divs with class containing "views-row"
        rows = soup.find_all('div', class_=lambda x: x and 'views-row' in str(x))
        
        print(f"Found {len(rows)} rows on page {page + 1}")
        
        for row in rows:
            # Look for the "Inaugural Address" link in this row
            links = row.find_all('a', href=True)
            
            for link in links:
                href = link.get('href', '')
                link_text = link.get_text(strip=True)
                
                # Check if this is an inaugural address link
                if link_text and ('inaugural' in link_text.lower() or 'address' in link_text.lower()):
                    # Get the date from the row
                    date_elem = row.find('span', class_=lambda x: x and 'date' in str(x).lower())
                    date = date_elem.get_text(strip=True) if date_elem else "Unknown date"
                    
                    # Get the president name
                    president_link = row.find('a', href=lambda x: x and '/people/' in str(x))
                    president = president_link.get_text(strip=True) if president_link else "Unknown president"
                    
                    # Make absolute URL if relative
                    if href.startswith('/'):
                        full_url = urljoin(BASE_URL, href)
                    else:
                        full_url = href
                    
                    addresses.append({
                        'url': full_url,
                        'title': link_text,
                        'date': date,
                        'president': president
                    })
                    break  # Only take one link per row
        
        all_addresses.extend(addresses)
        
        # Check if there are more pages
        if len(rows) == 0:
            print(f"No more rows on page {page + 1}, stopping pagination")
            break
        
        # Small delay between pages
        time.sleep(1)
    
    print(f"Found {len(all_addresses)} total inaugural address links")
    
    return all_addresses

def scrape_single_address(url, title, date="Unknown date", president="Unknown president"):
    """Scrape a single inaugural address."""
    print(f"\nScraping: {title}")
    print(f"URL: {url}")
    
    response = get_page_with_retry(url)
    
    if not response:
        print(f"Failed to fetch: {url}")
        return None
    
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Extract the address text
    text = extract_address_text(soup)
    
    if not text:
        print(f"Warning: No text found for {title}")
        return None
    
    return {
        'url': url,
        'title': title,
        'president': president,
        'date': date,
        'text': text
    }

def save_address(address_data, output_dir="txt"):
    """Save an inaugural address to a file."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create a safe filename from the president and date
    # Extract year from date first
    year_match = re.search(r'\d{4}', address_data['date'])
    year = year_match.group(0) if year_match else "unknown"
    
    # Create safe president name
    safe_president = re.sub(r'[^\w\s-]', '', address_data['president'])
    safe_president = re.sub(r'\s+', '_', safe_president)[:30]  # Limit length
    
    # Create filename
    filename = f"{year}_{safe_president}.txt"
    
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        # Write just the text, no metadata (for text analysis)
        f.write(address_data['text'])
    
    return filepath

def export_to_csv(all_addresses, filename="inaugural_addresses.csv"):
    """Export all addresses to a CSV file."""
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['title', 'president', 'date', 'url', 'text_length'])
        writer.writeheader()
        
        for addr in all_addresses:
            writer.writerow({
                'title': addr['title'],
                'president': addr['president'],
                'date': addr['date'],
                'url': addr['url'],
                'text_length': len(addr['text'])
            })
    
    print(f"\nExported metadata to {filename}")

def main():
    """Main scraping function."""
    print("INAUGURAL ADDRESS SCRAPER")
    print("American Presidency Project")
    print("="*70)
    
    # Get list of all addresses
    address_links = get_address_links()
    
    if not address_links:
        print("No addresses found!")
        return
    
    # Scrape each address
    all_addresses = []
    
    for i, link_info in enumerate(address_links, 1):
        print(f"\n[{i}/{len(address_links)}]", end=" ")
        
        address_data = scrape_single_address(
            link_info['url'], 
            link_info['title'],
            date=link_info.get('date', 'Unknown date'),
            president=link_info.get('president', 'Unknown president')
        )
        
        if address_data:
            # Save individual file
            filepath = save_address(address_data)
            all_addresses.append(address_data)
            
            print(f"Saved to {filepath}")
        
        # Be respectful - add a small delay between requests
        time.sleep(1)
    
    # Export to CSV
    if all_addresses:
        export_to_csv(all_addresses)
        print(f"\n{'='*70}")
        print(f"Successfully scraped {len(all_addresses)} inaugural addresses")
        print(f"Individual files saved to: inaugural_addresses/")
        print(f"Metadata exported to: inaugural_addresses.csv")
    else:
        print("\nNo addresses were successfully scraped")

if __name__ == "__main__":
    main()

