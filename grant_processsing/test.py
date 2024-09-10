import inflect

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
    
print(word_to_number('one'))
print(word_to_number('two'))
print(word_to_number('three'))
print(word_to_number('four'))
print(word_to_number('five'))