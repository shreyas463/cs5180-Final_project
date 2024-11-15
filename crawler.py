import urllib.request
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import pymongo
from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)


class Frontier:
    def __init__(self):
        self.queue = []
        self.visited = set()
        self.done_flag = False

    def addURL(self, url):
        if url not in self.visited and url not in self.queue:
            self.queue.append(url)

    def nextURL(self):
        if self.queue:
            url = self.queue.pop(0)
            self.visited.add(url)
            return url
        else:
            return None

    def done(self):
        return self.done_flag or not self.queue

    def clear_frontier(self):
        self.queue = []
        self.done_flag = True


class FacultySearch:
    def __init__(self, collection):
        """Initialize search engine with MongoDB collection"""
        self.collection = collection
        self.documents = []
        self.urls = []
        self.vectorizer = None
        self.tfidf_matrix = None
        self.initialize_search_index()

    def initialize_search_index(self):
        """Process faculty pages and build search index"""
        logging.info("Initializing search index...")
        self.documents, self.urls = self._process_faculty_pages()
        if self.documents:
            self._build_index()

    def _process_faculty_pages(self):
        """Process faculty pages to extract searchable content"""
        faculty_pages = self.collection.find({'is_faculty': True})
        documents = []
        urls = []

        for page in faculty_pages:
            content = self._extract_faculty_content(page)
            if content:
                documents.append(content)
                urls.append(page['url'])

        logging.info(f"Processed {len(documents)} faculty pages")
        return documents, urls

    def _extract_faculty_content(self, page):
        """Extract relevant content from faculty page"""
        soup = BeautifulSoup(page['html'], 'html.parser')

        # Extract from fac-info div
        fac_info = soup.find('div', class_='fac-info')
        if not fac_info:
            # Try to extract content from the entire page if fac-info not found
            content = soup.get_text(separator=' ', strip=True)
            return content if content else None

        # Get faculty details
        name = self._get_text(fac_info.find('h1'))
        title = self._get_text(fac_info.find('h2'))

        # Get research content
        research_content = self._extract_research_content(soup)

        return f"{name} {title}: {research_content}"

    def _get_text(self, element):
        """Safely extract text from BS4 element"""
        return element.get_text(strip=True) if element else ''

    def _extract_research_content(self, soup):
        """Extract research-related content from page"""
        # Try dedicated research area div
        research_div = soup.find('div', class_='research-areas')
        if research_div:
            return research_div.get_text(separator=' ', strip=True)

        # Try research-related headings
        research_content = []
        research_headings = soup.find_all(
            ['h2', 'h3', 'h4'],
            string=lambda x: x and any(term in x.lower() for term in [
                'research', 'interests', 'expertise'])
        )

        for heading in research_headings:
            content = []
            for sibling in heading.next_siblings:
                if sibling.name and sibling.name.startswith('h'):
                    break
                if hasattr(sibling, 'get_text'):
                    content.append(sibling.get_text(strip=True))
            research_content.append(' '.join(content))

        # If no research content found, extract all text content
        if not research_content:
            main_content = soup.find('main') or soup.find(
                'div', class_='main-content')
            if main_content:
                return main_content.get_text(separator=' ', strip=True)
            return soup.get_text(separator=' ', strip=True)

        return ' '.join(research_content)

    def _build_index(self):
        """Build TF-IDF search index"""
        # Modified parameters to handle small document collections
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_df=1.0,
            min_df=1,
            strip_accents='unicode'
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)

    def search(self, query, page=0, page_size=5):
        """
        Search faculty pages and return paginated results
        Returns: (results, pagination_info)
        """
        if not self.documents:
            return [], {'total': 0, 'current_page': 0, 'total_pages': 0}

        try:
            # Get search results
            query_vec = self.vectorizer.transform([query])
            similarities = linear_kernel(
                query_vec, self.tfidf_matrix).flatten()

            # Sort results by relevance
            relevant_indices = similarities.argsort()[::-1]
            relevant_indices = [
                idx for idx in relevant_indices if similarities[idx] > 0]

            # Calculate pagination
            total_results = len(relevant_indices)
            total_pages = (total_results + page_size - 1) // page_size
            start_idx = page * page_size
            end_idx = min(start_idx + page_size, total_results)

            # Format results
            results = []
            for idx in relevant_indices[start_idx:end_idx]:
                results.append({
                    'url': self.urls[idx],
                    'score': similarities[idx],
                    'snippet': self._create_snippet(self.documents[idx], query)
                })

            pagination_info = {
                'total': total_results,
                'current_page': page,
                'total_pages': total_pages
            }

            return results, pagination_info
        except Exception as e:
            logging.error(f"Search error: {e}")
            return [], {'total': 0, 'current_page': 0, 'total_pages': 0}

    def _create_snippet(self, text, query, max_length=200):
        """Create a relevant text snippet highlighting query terms"""
        query_terms = query.lower().split()
        text_lower = text.lower()

        # Find best snippet starting position
        start_pos = 0
        for term in query_terms:
            pos = text_lower.find(term)
            if pos != -1:
                start_pos = max(0, pos - 50)
                break

        # Create snippet
        snippet = text[start_pos:start_pos + max_length]
        if len(text) > max_length:
            snippet += '...'

        return snippet


def extract_faculty_details(html):
    soup = BeautifulSoup(html, 'html.parser')

    details = {
        'email': None,
        'phone_number': None,
        'location': None,
        'office_hours': None
    }

    # Extract email
    email_tag = soup.find('p', class_='emailicon')
    if email_tag:
        details['email'] = email_tag.get_text(strip=True)

    # Extract phone number
    phone_tag = soup.find('p', class_='phoneicon')
    if phone_tag:
        details['phone_number'] = phone_tag.get_text(strip=True)

    # Extract location
    location_tag = soup.find('p', class_='locationicon')
    if location_tag:
        details['location'] = location_tag.get_text(strip=True)

    # Extract office hours
    office_hours_tag = soup.find('p', class_='hoursicon')
    if office_hours_tag:
        details['office_hours'] = office_hours_tag.get_text(strip=True)

    return details


def retrieveURL(url):
    try:
        with urllib.request.urlopen(url) as response:
            html = response.read()
            return html
    except Exception as e:
        logging.error(f"Failed to retrieve {url}: {e}")
        return None


def storePage(url, html, is_faculty, collection):
    page = {
        'url': url,
        'html': html.decode('utf-8', errors='ignore'),
        'is_faculty': is_faculty
    }

    if is_faculty:
        # Extract additional details if it's a faculty page
        faculty_details = extract_faculty_details(html)
        page.update(faculty_details)

    collection.insert_one(page)


def parse(html, base_url):
    soup = BeautifulSoup(html, 'html.parser')
    links = []
    for link in soup.find_all('a', href=True):
        href = link['href']
        full_url = urljoin(base_url, href)
        # Only add URLs within the same domain
        if urlparse(full_url).netloc == urlparse(base_url).netloc:
            links.append(full_url)
    return soup, links


def target_page(soup):
    # Identify faculty profile pages by checking for the 'fac-info' class
    fac_info_div = soup.find('div', class_='fac-info')
    if fac_info_div:
        return True
    return False


def crawlerThread(frontier, num_targets, collection):
    targets_found = 0
    while not frontier.done():
        url = frontier.nextURL()
        if url is None:
            break
        logging.info(f"Processing URL: {url}")
        html = retrieveURL(url)
        if html is None:
            continue
        soup, new_links = parse(html, url)
        is_faculty = target_page(soup)
        storePage(url, html, is_faculty, collection)
        if is_faculty:
            targets_found += 1
            logging.info(f"Found faculty page: {
                         url} ({targets_found}/{num_targets})")
        if targets_found >= num_targets:
            frontier.clear_frontier()
        else:
            for new_url in new_links:
                if new_url not in frontier.visited:
                    frontier.addURL(new_url)


def run_search_interface(search_engine):
    """Run interactive search interface"""
    print("\nWelcome to the Faculty Research Search Engine")
    print("===========================================")

    while True:
        # Get search query
        query = input("\nEnter search query (or 'exit' to quit): ").strip()
        if query.lower() == 'exit':
            break

        # Initial search
        page = 0
        while True:
            # Get results for current page
            results, pagination = search_engine.search(query, page=page)

            # Display results
            if not results:
                print("\nNo results found.")
                break

            print(f"\nShowing results {page * 5 + 1} to {min((page + 1) * 5, pagination['total'])} "
                  f"of {pagination['total']} for '{query}'\n")

            for result in results:
                print(f"URL: {result['url']}")
                print(f"Relevance Score: {result['score']:.4f}")
                print(f"Snippet: {result['snippet']}")
                print("-" * 60)

            # Handle pagination
            if pagination['total_pages'] > 1:
                print(f"\nPage {page + 1} of {pagination['total_pages']}")
                if page > 0:
                    print("'b' for previous page", end=' ')
                if page < pagination['total_pages'] - 1:
                    print("'n' for next page", end=' ')
                print("'q' for new search", end=' ')

                choice = input("\nChoice: ").lower()
                if choice == 'n' and page < pagination['total_pages'] - 1:
                    page += 1
                elif choice == 'b' and page > 0:
                    page -= 1
                elif choice == 'q':
                    break
                else:
                    print("Invalid choice. Please try again.")
            else:
                input("\nPress Enter for a new search...")
                break


def main():
    # Set up MongoDB connection
    client = MongoClient('localhost', 27017)
    db = client['mydatabase']
    collection = db['pages']

    # Crawling phase
    print("Starting crawler...")
    seed_url = 'https://www.cpp.edu/sci/biological-sciences/index.shtml'
    num_targets = 10
    frontier = Frontier()
    frontier.addURL(seed_url)
    crawlerThread(frontier, num_targets, collection)
    print("Crawling completed.")

    # Search phase
    print("\nInitializing search engine...")
    search_engine = FacultySearch(collection)
    run_search_interface(search_engine)


if __name__ == '__main__':
    main()


# to test ex:  try cell biology or environmental science or etc
