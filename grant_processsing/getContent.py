# see view-source:https://grants.nih.gov/grants/guide/rfa-files/RFA-AG-12-012.html#_Section_IV._Application_1
# https://grants.nih.gov/grants/guide/rfa-files/RFA-AG-12-012.html
# https://grants.nih.gov/grants/guide/rfa-files/RFA-HL-12-004.html
# view-source:https://grants.nih.gov/grants/guide/rfa-files/RFA-ES-15-007.html#_Section_IV._Application_1
# https://grants.gov/
# https://www.grants.gov/search-grants
# https://pubmed.ncbi.nlm.nih.gov/21028883/
# https://api.reporter.nih.gov/?urls.primaryName=V2.0
# https://reporter.nih.gov/exporter/projects

import requests
from bs4 import BeautifulSoup
import re
import inflect
import spacy
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English

def extract_webpage_sections(url):#beter pandas gebruiken
    
    if url == 'https://grants.nih.gov/grants/guide/rfa-files/RFA-AA-12-002.html':
       print('test')
    # Send a GET request to the URL
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        webpage_content = soup.get_text(separator=' ')
        type = checkTypeOfTable(webpage_content)
        # voorwaarde: er moet een soort overzicht van algemene informatie zijn
        # https://grants.nih.gov/grants/guide/rfa-files/RFA-HD-02-031.html heeft geen overzicht van algemene informatie
        # https://grants.nih.gov/grants/guide/rfa-files/RFA-NS-01-012.html ook niet
        if type == 1:
            # take the part from the webpage_content starting from the first occurrence of 'Participating Organizations' and ending at the first time two newlines characters occur in a row after 'due dates'
            table_text = extract_overview_1(webpage_content)
        elif type == 2:
            table_text = extract_overview_2(webpage_content)
        elif type == 3:
            table_text = extract_overview_3(webpage_content)
        elif type == 4:
            table_text = extract_overview_4(webpage_content)
        if type == 0:
            raise ValueError("Does not contain some kind of overview of the RFA.")
        if len(table_text) < 100:
            raise ValueError("Table content too short or empty.")
    else:
        raise ValueError(f"Failed to retrieve the webpage. Status code: {response.status_code}")
    webpage_content = re.sub(r'https?://\S+', '', webpage_content)
    if type == 1 or type == 2:
        # Use a regular expression to handle variations in whitespace and line breaks
        part2_occurrences = re.split(r'Part\s*2', webpage_content, 2, flags=re.IGNORECASE)
        if(len(part2_occurrences) <= 1):
            part2_occurrences = re.split(r'Part\s*II', webpage_content, 2, flags=re.IGNORECASE)

        if len(part2_occurrences) <= 1:
            raise ValueError("Neither 'Part 2' nor 'Part II' found on the webpage.")

       # Extract the content after the second occurrence
        try:
            webpage_content = part2_occurrences[2]
        except:
            webpage_content = part2_occurrences[1]
        webpage_content = process_text(webpage_content)
    elif type == 4:
        part2_occurrences = webpage_content.lower().split('eligible institutions', 2)
        if len(part2_occurrences) >= 2:
        # Select the content after the second occurrence of 'Part 2'
          webpage_content = 'eligible institutions' + part2_occurrences[2]
        else:
          raise ValueError("Part 2 not found on the webpage.")
    elif type == 3:
        part2_occurrences = webpage_content.lower().split('purpose of this rfa', 2)
        if len(part2_occurrences) >= 2:
        # Select the content after the second occurrence of 'Part 2'
          webpage_content = 'PURPOSE OF THIS RFA' + part2_occurrences[2]
        else:
          raise ValueError("Part 2 not found on the webpage.")
    else:
       raise ValueError("Part 2 not found on the webpage.")
    if type == 1 or type == 2:
      sections = extract_sections1(webpage_content)
    elif type == 3:
      sections = extract_sections3(webpage_content)
    elif type == 4:
      sections = extract_sections4(webpage_content)
    else:
       raise ValueError("not implemented")
    section_ii_content = sections['section_ii_content']
    section_iii_content = sections['section_iii_content']
    # if section ii content < 100 characters, raise ValueError
    if len(section_ii_content) < 100:
        raise ValueError("Section II content too short or empty.")
    # if section iii is empty or too short, raise ValueError
    if len(section_iii_content) < 100:
        raise ValueError("Section III content too short or empty.")
    section_iv_content = sections['section_iv_content']
    if len(section_iv_content) < 100:
        raise ValueError("Section IV content too short or empty.")
    table = table_text
    return table, section_iii_content, section_iv_content, section_ii_content


def extract_sections1(text):
  section_ii_content = ""
  found_section_ii = False
  ii_done = False

  for line in text.splitlines():
    # Use regular expression to match section titles with potential variations:
    if re.match(r"(?:^\s*Section\s*II\.\s*|\s+II\.|^II\.)(.*)", re.sub(r"\r\n", "\n", line), re.IGNORECASE):
      found_section_ii = True
    if found_section_ii and re.search(r"III", re.sub(r"\r\n", "\n", line), re.IGNORECASE):
      ii_done = True
      break
    if found_section_ii and not ii_done:
      section_ii_content += line + "\n"

  section_iii_content = ""
  section_iv_content = ""
  found_section_iii = False
  found_section_iv = False
  iii_done = False
  iv_done = False

  for line in text.splitlines():
    # Use regular expression to match section titles with potential variations:
    if not iii_done and re.match(r"(?:^\s*Section\s*III\.\s*|\s+III\.|^III\.)(.*)", re.sub(r"\r\n", "\n", line), re.IGNORECASE):
      found_section_iii = True
    elif re.match(r"(?:^\s*Section\s*IV\.\s*|\s+IV\.|^IV\.)(.*)", re.sub(r"\r\n", "\n", line), re.IGNORECASE) and found_section_iii and not iv_done:
      found_section_iv = True
      iii_done = True
    elif re.match(r"(?:^\s*Section\s*V\.\s*|\s+V\.|^V\.)", re.sub(r"\r\n", "\n", line), re.IGNORECASE) and found_section_iv and not iv_done:
      iv_done = True
      break

    if found_section_iii and not iii_done:
      section_iii_content += line + "\n"
    elif found_section_iv and iii_done:
      section_iv_content += line + "\n"

  return {
      "section_ii_content": section_ii_content.strip(),
      "section_iii_content": section_iii_content.strip(),
      "section_iv_content": section_iv_content.strip(),
  }


def checkTypeOfTable(content):
   # if you can find 'Participating Organizations' in the content, and 'Due Dates' in the content
   # then the table is in format 1, return 1
  if 'Participating Organizations' in content and 'Due Dates' in content:
    return 1
  elif 'Participating Organization(s)' in content and 'Due Dates' in content:
    return 2
  elif re.search(r'\bparticipating organizations\b', content.lower()) and not 'Due Dates' in content:
    return 3
  elif re.search(r'participating organization:', content.lower()) and not 'Due Dates' in content:
    return 4
  else:
    return 0

def extract_overview_1(webpage_content):
    # Convert the content to lowercase for case-insensitive search
    content_lower = webpage_content.lower()
    # Find the start of the desired part
    start = content_lower.find('participating organizations')
    # Cut the string from the start position
    cut_content = webpage_content[start:]
    # Find the position of 'Due Dates'
    due_dates_pos = content_lower.find('due dates', start)
    # Find the end of the desired part after 'Due Dates'
    end = cut_content.find('\n\n', due_dates_pos - start)
    if end == -1:
        end = cut_content.find('\n \n', due_dates_pos - start)
        if end == -1:
           end = cut_content.find('.\r\n', due_dates_pos - start)
           if end == -1:
              return 'Two newline characters in a row not found in cut webpage content after Due Dates'
    # Extract the desired part
    part = cut_content[:end]
    return part

def extract_overview_2(webpage_content):
    # Convert the content to lowercase for case-insensitive search
    content_lower = webpage_content.lower()
    # Find the start of the desired part
    start = content_lower.find("participating organization(s)")
    # Cut the string from the start position
    cut_content = webpage_content[start:]
    # Find the position of 'Due Dates'
    due_dates_pos = content_lower.find('due dates', start)
    # Find the end of the desired part after 'Due Dates'
    end = cut_content.find('\n\n', due_dates_pos - start)
    if end == -1:
        end = cut_content.find('\n \n', due_dates_pos - start)
        if end == -1:
          return 'Two newline characters in a row not found in cut webpage content after Due Dates'
    # Extract the desired part
    part = cut_content[:end]
    return part

def extract_overview_3(webpage_content):
    # Convert the content to lowercase for case-insensitive search
    match = re.search(r'(participating organizations[:\n\r]*)(.*?)(application receipt date[:\n\r]*)(.*?)(\r\n|\r\r|\n\n|\n\r)', webpage_content, re.DOTALL | re.IGNORECASE)
    # Check if a match was found
    if match:
        # Extract the part of the string between the start and end strings
        part = match.group(2)
    else:
        part = 'Start or end string not found in content'
    return part
"""
def extract_overview_3(webpage_content):
    # Convert the content to lowercase for case-insensitive search
    match = re.search(r'(participating organization:[:\n\r]*)(.*?)(application receipt date[:\n\r]*)(.*?)(\r\n|\r\r|\n\n|\n\r)', webpage_content, re.DOTALL | re.IGNORECASE)
    # Check if a match was found
    if match:
        # Extract the part of the string between the start and end strings
        part = match.group(2)
    else:
        part = extract_overview_3(webpage_content)
    return part"""

def extract_overview_4(webpage_content):
    start = webpage_content.lower().find('participating organization:')
    end = webpage_content.lower().find('application receipt date:')
    date_end = webpage_content[end:].find('\r\n', end)
    result = webpage_content[start:date_end]
    return result
def process_text(input_text):
    # Split the text into lines
    lines = input_text.split('\n')

    # Check if the word 'Section' occurs 3 or more times in the first 3 lines
    if sum(line.lower().count('section') for line in lines[:8]) >= 3:
        # Find the index where 'Part 2' or 'Part II' occurs (case insensitive)
        part_index = next((i for i, line in enumerate(lines) if re.search(r'part\s*2|part\s*II', line, re.IGNORECASE)), -1)

        if part_index != -1:
            # Remove everything before 'Part 2' or 'Part II'
            lines = lines[part_index:]

    # Join the lines back into a string
    result_text = '\n'.join(lines)

    return result_text

def extract_sections3(text):
    # Find the part between 'eligible institutions' and 'peer review process' (case insensitive)
    start_pattern = re.compile(r'eligible\s*\S*\s*institutions', re.IGNORECASE)
    end_pattern = re.compile(r'peer\s*review\s*process', re.IGNORECASE)

    start_match = start_pattern.search(text)
    end_match = end_pattern.search(text)

    if start_match and end_match:
        part_between = text[start_match.end():end_match.start()]

        # Split the part into two sections based on the first occurrence of 'WHERE TO SEND INQUIRIES' (case insensitive)
        inquiries_pattern = re.compile(r'where\s*\S*\s*to\s*send\s*inquiries', re.IGNORECASE)
        inquiries_match = inquiries_pattern.search(part_between)

        if inquiries_match:
            section_iii = part_between[:inquiries_match.start()]
            section_iv = part_between[inquiries_match.end():]
            return {
                  "section_iii_content": section_iii,
                  "section_iv_content": section_iv,
                }

    return None, None



def extract_sections4(text):
    """
    Extracts content from two specific sections of a given text, handling potential whitespace variations,
    non-breaking spaces, and potential variations in section formatting.

    Args:
      text: The string of text to search.

    Returns:
      A dictionary containing two keys:
        - "section_iii": The content of Section III.
        - "section_iv": The content of Section IV.
    """

    section_iii_content = ""
    section_iv_content = ""
    found_section_iii = False
    found_section_iv = False
    iii_done = False
    iv_done = False

    # Find the part between 'eligible institutions' and the last 'peer review process'
    start_pattern = re.compile(r'eligible\s*\S*\s*institutions', re.IGNORECASE)
    end_pattern = re.compile(r'peer\s*review\s*process', re.IGNORECASE)

    start_match = start_pattern.search(text)
    end_matches = [match.start() for match in end_pattern.finditer(text)]

    if start_match and end_matches:
        # Find the last occurrence of 'peer review process'
        end_position = end_matches[-1]
        part_between = text[start_match.end():end_position]

        # Split the part into two sections based on the first occurrence of 'WHERE TO SEND INQUIRIES' (case insensitive)
        inquiries_pattern = re.compile(r'where\s*\S*\s*to\s*send\s*inquiries', re.IGNORECASE)
        inquiries_match = inquiries_pattern.search(part_between)

        if inquiries_match:
            section_ii_content = part_between[:inquiries_match.start()]
            section_iii_content = part_between[:inquiries_match.start()]
            section_iv_content = part_between[inquiries_match.end():]

    return {
        "section_ii_content": section_ii_content.strip(),
        "section_iii_content": section_iii_content.strip(),
        "section_iv_content": section_iv_content.strip(),
    }



def clean_text(text):
    # Replace multiple whitespaces with a single whitespace
    cleaned_text = ' '.join(text.split())

    # Remove commas or dots from numbers
    cleaned_text = re.sub(r'\$([\d,.]+)', lambda match: match.group(0).replace(',', '').replace('.', ''), cleaned_text)

    return cleaned_text
""" 
def custom_tokenizer(nlp):
    infix_re = spacy.util.compile_infix_regex(nlp.Defaults.infixes + (r'(?<=[0-9])\.(?=[0-9])',))
    return Tokenizer(nlp.vocab, infix_finditer=infix_re.finditer)


def extract_budget_and_years(text):
    nlp = spacy.load("en_core_web_sm")
    
    # Process the text with spaCy
    doc = nlp(text)

    # Initialize variables to store budget per year and number of years
    budget_per_year = None
    years = None

    # Iterate through each sentence in the document
    for sentence in doc.sents:
        # Check if the sentence contains keywords related to budget and years
        if "costs" in sentence.text.lower() and ("year" in sentence.text.lower() or "years" in sentence.text.lower()):
            # Extract the budget per year
            nlp.tokenizer = custom_tokenizer(nlp)
            for token in sentence:
                if token.pos_ == "NUM":
                    budget_per_year = float(token.text.replace(",", ""))
                    break
        
        if "years" in sentence.text.lower():
            nlp = spacy.load("en_core_web_sm")
            # Extract the number of years
            for token in sentence:
                if token.pos_ == "NUM":
                    numba = token.text
                    try:
                        years = int(numba)
                    except:
                        years = word_to_number(numba)
                    break

    return budget_per_year, years

def word_to_number(word):
    # Define the mapping dictionary
    number_map = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, 
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, 
        "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, 
        "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, 
        "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, 
        "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90
    }
    
    # Handle larger numbers
    multipliers = {
        "hundred": 100,
        "thousand": 1000,
        "million": 1000000,
        "billion": 1000000000
    }
    
    # Split the input into words
    words = word.lower().replace('-', ' ').split()
    current = 0
    total = 0
    
    for w in words:
        if w in number_map:
            current += number_map[w]
        elif w in multipliers:
            current *= multipliers[w]
            total += current
            current = 0
        elif w == "and":
            continue
        else:
            return None
    
    return total + current

"""